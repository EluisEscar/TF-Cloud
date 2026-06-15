# Pruebas de rendimiento, escalabilidad y disponibilidad

Scripts para medir cómo se comporta la API en EC2 bajo condiciones reales.
Los resultados generados (JSON en `tests/results/`) son evidencia directa
para la sección de "Pruebas" del informe.

## Setup

```bash
pip install requests
```

Configura la URL de la API (puede ir en `.env`):

```bash
export API_URL=http://<TU_IP_EC2>:8000
```

## Tipos de prueba

### 1. Load test — comportamiento bajo carga sostenida

Simula N usuarios concurrentes pegándole a la API sin pausa durante T segundos.
Mide latencia y throughput en régimen estable.

```bash
python tests/load_test.py --mode load --concurrent 10 --duration 60
```

**Interpretación:**
- `p50`: latencia mediana — el "usuario típico"
- `p95`: el 5 % más lento — usuarios en el peor escenario habitual
- `p99`: outliers, casos raros (GC, network jitter)
- `Throughput (RPS)`: cuántas predicciones por segundo aguanta

**Valores esperables para `t3.micro` + sklearn 1.6 + modelo de 124 MB:**
| Métrica | Valor típico |
|---|---|
| p50 latencia | 40-80 ms |
| p95 latencia | 100-200 ms |
| p99 latencia | 200-500 ms |
| Throughput | 20-50 RPS |

### 2. Stress test — punto de saturación

Aumenta gradualmente la concurrencia hasta que el sistema empieza a fallar
(error rate > 5 %). Encuentra el límite operativo.

```bash
python tests/load_test.py --mode stress --max-concurrent 200 --step 10 --duration 20
```

El script imprime una tabla:
```
  → Concurrencia   10 ... RPS= 45.2  p50= 60ms  p95=120ms  err=0.0%
  → Concurrencia   20 ... RPS= 71.8  p50= 80ms  p95=180ms  err=0.0%
  → Concurrencia   50 ... RPS=120.5  p50=180ms  p95=450ms  err=0.5%
  → Concurrencia  100 ... RPS=140.2  p50=350ms  p95=900ms  err=4.8%
  → Concurrencia  150 ... RPS=130.1  p50=520ms  p95=1500ms err=12.3%
  ⚠️  Error rate >5% → punto de saturación encontrado.
```

**Interpretación:**
- El punto donde el error rate cruza 5 % es la **capacidad máxima**.
- Antes del cruce, latencias crecen pero el sistema responde.
- Después del cruce, requests empiezan a fallar (timeouts, conexiones rechazadas).

### 3. Availability test — uptime sostenido

Mide cuántos requests exitosos hay en una ventana larga (típicamente
10 min a varias horas). Métrica clave: **% de uptime**.

```bash
python tests/load_test.py --mode availability --duration 600 --interval 5
```

Esto manda 1 request cada 5 segundos durante 10 minutos = 120 requests.
Si todos pasan → uptime 100 %.

**Valores aceptables:**
- `≥ 99.9 %` (3 nueves): producción
- `≥ 99 %`  (2 nueves): tier free razonable
- `< 99 %`: revisar infraestructura

## Resultados

Cada corrida guarda un JSON en `tests/results/` con timestamp en el nombre:

```
tests/results/load_10c_60s_20260615-220300.json
tests/results/stress_200max_20260615-220545.json
tests/results/availability_600s_20260615-221012.json
```

El JSON tiene la estructura completa: configuración usada + métricas. Ideal
para incluir en el informe (anexo) o graficar después.

## Cómo usarlo para el informe

Sección sugerida del informe:

### 1. Setup de pruebas
- Herramienta: script Python con `requests` + `ThreadPoolExecutor`.
- Endpoint medido: `POST /predict` (la operación crítica).
- Hardware: EC2 t3.micro (1 vCPU, 1 GB RAM).
- Modelo: RandomForest 124 MB cargado en memoria.

### 2. Load test (rendimiento normal)
Reportar: tabla con latencias p50/p95/p99 + RPS + 1 línea de conclusión.

### 3. Stress test (escalabilidad)
Reportar: la tabla completa que imprime el script + cuál es el punto de
saturación + recomendación (por ej. "para >100 concurrentes considerar
t3.small o Auto Scaling Group").

### 4. Availability test (disponibilidad)
Reportar: % de uptime + total de requests + caídas (si hubo).

### 5. Conclusiones
Comparar contra SLA objetivo (ej. "el sistema cumple con 99 % uptime y
latencia p95 < 200 ms bajo carga normal").

## Notas operativas

- **No corras stress test contra la EC2 si Streamlit Cloud está activamente
  consultando** — vas a degradar la app real durante la prueba.
- **El primer request siempre es más lento** (cold start del modelo en RAM).
  Si te interesa medir steady-state, descarta los primeros 5 segundos del
  load test.
- **Free Tier de AWS no es ilimitado**: un stress muy agresivo puede consumir
  más banda de la cuota gratuita. Mantenelo en ventanas cortas (1-2 min).
