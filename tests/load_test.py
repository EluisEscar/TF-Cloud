"""Pruebas de rendimiento, escalabilidad y disponibilidad de la API en EC2.

Genera métricas requeridas para el informe (sección "Pruebas"):
- Tiempo de respuesta y latencia (p50, p95, p99)
- Throughput (requests por segundo)
- Tasa de error
- Uptime

Modos:
- ``load``         carga sostenida con N hilos concurrentes durante T segundos
- ``stress``       rampa creciente de concurrencia hasta saturación
- ``availability`` polling periódico durante T segundos para medir uptime

Ejemplos:
    # Carga normal: 10 usuarios concurrentes durante 1 min
    python tests/load_test.py --mode load --concurrent 10 --duration 60

    # Stress: aumenta de 10 en 10 hasta 200 (o hasta que falle >5%)
    python tests/load_test.py --mode stress --max-concurrent 200 --step 10

    # Disponibilidad: 1 request cada 5s durante 10 min
    python tests/load_test.py --mode availability --duration 600 --interval 5

Variables de entorno:
    API_URL  URL base de la API (default http://localhost:8000)
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

# Payload realista para /predict — estación de Dhaka, Bangladesh
PAYLOAD = {
    "pm1": 30.0,
    "relativehumidity": 65.0,
    "temperature": 28.0,
    "um003": 3000.0,
    "hour": 14,
    "day": 15,
    "month": 6,
    "year": 2026,
    "dayofweek": 4,
    "lat": 23.73,
    "lon": 90.40,
    "country_code": "BD",
}


def call_predict(api_url: str, timeout: float = 10.0) -> tuple[bool, float, int]:
    """Hace POST /predict. Devuelve (success, latency_ms, status_code)."""
    start = time.perf_counter()
    status = 0
    try:
        r = requests.post(f"{api_url}/predict", json=PAYLOAD, timeout=timeout)
        status = r.status_code
        ok = status == 200
    except requests.exceptions.RequestException:
        ok = False
    latency_ms = (time.perf_counter() - start) * 1000
    return ok, latency_ms, status


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(results: list[tuple[bool, float, int]], duration: float | None = None) -> dict:
    latencies = [lat for ok, lat, _ in results if ok]
    errors = sum(1 for ok, _, _ in results if not ok)
    total = len(results)
    err_rate = (errors / total) if total else 0.0
    return {
        "total_requests": total,
        "successful": total - errors,
        "failed": errors,
        "error_rate_pct": round(100 * err_rate, 3),
        "throughput_rps": round(total / duration, 2) if duration else None,
        "latency_ms": {
            "min": round(min(latencies), 2) if latencies else 0,
            "mean": round(statistics.mean(latencies), 2) if latencies else 0,
            "p50": round(statistics.median(latencies), 2) if latencies else 0,
            "p95": round(percentile(latencies, 95), 2),
            "p99": round(percentile(latencies, 99), 2),
            "max": round(max(latencies), 2) if latencies else 0,
        },
    }


def print_summary(s: dict, title: str) -> None:
    bar = "═" * 64
    print(f"\n{bar}\n  {title}\n{bar}")
    print(f"  Total requests   : {s['total_requests']:>8,}")
    print(f"  Successful       : {s['successful']:>8,}  ({100 - s['error_rate_pct']:.2f} %)")
    print(f"  Failed           : {s['failed']:>8,}  ({s['error_rate_pct']:.2f} %)")
    if s.get("throughput_rps") is not None:
        print(f"  Throughput       : {s['throughput_rps']:>8.1f} RPS")
    lat = s["latency_ms"]
    print(f"  Latencia (ms)")
    print(f"      min / p50 / p95 / p99 / max  =  "
          f"{lat['min']} / {lat['p50']} / {lat['p95']} / {lat['p99']} / {lat['max']}")
    print(f"      mean = {lat['mean']}")
    print(bar)


def run_load_test(api_url: str, concurrent: int, duration: float) -> list[tuple[bool, float, int]]:
    """N hilos pegándole sin pausa durante T segundos."""
    results: list[tuple[bool, float, int]] = []
    stop_at = time.time() + duration

    def worker() -> list[tuple[bool, float, int]]:
        local = []
        while time.time() < stop_at:
            local.append(call_predict(api_url))
        return local

    with ThreadPoolExecutor(max_workers=concurrent) as ex:
        futures = [ex.submit(worker) for _ in range(concurrent)]
        for f in as_completed(futures):
            results.extend(f.result())
    return results


def run_stress_test(
    api_url: str,
    max_concurrent: int,
    step: int,
    duration_per_step: float,
    error_threshold: float = 0.05,
) -> list[dict]:
    """Rampa: 10 → 20 → 30 ... hasta saturación o max_concurrent."""
    history = []
    for c in range(step, max_concurrent + 1, step):
        print(f"  → Concurrencia {c:>4} ...", end=" ", flush=True)
        results = run_load_test(api_url, c, duration_per_step)
        s = summarize(results, duration_per_step)
        history.append({"concurrent": c, **s})
        print(
            f"RPS={s['throughput_rps']:>6.1f}  "
            f"p50={s['latency_ms']['p50']:>5.0f}ms  "
            f"p95={s['latency_ms']['p95']:>5.0f}ms  "
            f"err={s['error_rate_pct']:>4.1f}%"
        )
        if s["error_rate_pct"] / 100 > error_threshold:
            print(f"\n  ⚠️  Error rate >{error_threshold:.0%} → punto de saturación encontrado.")
            break
    return history


def run_availability_test(
    api_url: str, duration: float, interval: float,
) -> list[tuple[bool, float, int]]:
    """Polling cada N segundos durante T segundos. Mide uptime."""
    results = []
    stop_at = time.time() + duration
    while time.time() < stop_at:
        results.append(call_predict(api_url))
        time.sleep(interval)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=["load", "stress", "availability"], required=True)
    parser.add_argument("--api-url", default=os.getenv("API_URL", "http://localhost:8000"))
    parser.add_argument("--concurrent", type=int, default=10)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--max-concurrent", type=int, default=100)
    parser.add_argument("--step", type=int, default=10)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--out-dir", type=Path, default=Path("tests/results"))
    args = parser.parse_args()

    api_url = args.api_url.rstrip("/")
    print(f"API URL: {api_url}")
    print(f"Mode   : {args.mode}")

    # Healthcheck previo — si la API no responde, no tiene sentido correr.
    try:
        r = requests.get(f"{api_url}/", timeout=5)
        r.raise_for_status()
    except Exception as e:
        print(f"\n❌ La API no responde en {api_url}: {e}")
        print("   Asegúrate que la EC2 está corriendo y el puerto 8000 está abierto.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    if args.mode == "load":
        print(f"\nLanzando load test: {args.concurrent} concurrentes × {args.duration}s ...")
        results = run_load_test(api_url, args.concurrent, args.duration)
        summary = {
            "mode": "load",
            "config": {"concurrent": args.concurrent, "duration_s": args.duration},
            **summarize(results, args.duration),
        }
        print_summary(summary, f"Load Test · {args.concurrent} concurrentes · {args.duration}s")
        out_file = args.out_dir / f"load_{args.concurrent}c_{args.duration}s_{ts}.json"

    elif args.mode == "stress":
        print(f"\nLanzando stress test: rampa de {args.step} → {args.max_concurrent} (step={args.step}, {args.duration}s/step) ...\n")
        history = run_stress_test(api_url, args.max_concurrent, args.step, args.duration)
        summary = {"mode": "stress", "config": vars(args), "history": history}
        # Punto de saturación = último que superó el threshold
        saturated_at = next(
            (h["concurrent"] for h in history if h["error_rate_pct"] / 100 > 0.05),
            None,
        )
        if saturated_at:
            print(f"\n  ★ Saturación detectada a partir de {saturated_at} concurrentes.")
        else:
            print(f"\n  ✅ Sin saturación hasta {args.max_concurrent} concurrentes.")
        out_file = args.out_dir / f"stress_{args.max_concurrent}max_{ts}.json"

    elif args.mode == "availability":
        print(f"\nLanzando availability test: 1 request cada {args.interval}s durante {args.duration}s ...")
        results = run_availability_test(api_url, args.duration, args.interval)
        summary = {
            "mode": "availability",
            "config": {"duration_s": args.duration, "interval_s": args.interval},
            **summarize(results, args.duration),
        }
        # Uptime explícito
        uptime_pct = (
            100 * summary["successful"] / summary["total_requests"]
            if summary["total_requests"] else 0
        )
        summary["uptime_pct"] = round(uptime_pct, 3)
        print_summary(summary, f"Availability Test · {args.duration}s · interval {args.interval}s")
        print(f"\n  ★ Uptime: {uptime_pct:.3f} %\n")
        out_file = args.out_dir / f"availability_{args.duration}s_{ts}.json"

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"📊 Resultados guardados en: {out_file}\n")


if __name__ == "__main__":
    main()
