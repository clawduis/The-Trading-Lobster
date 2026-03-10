# 🦞 The Trading Lobster

> A lobster's neural activity — translated into live BTC perpetual positions on Hyperliquid.

The Trading Lobster captures real-time electrical signals from a lobster's stomatogastric ganglion (STG), processes them through a spike detection and classification pipeline, and executes long or short BTC trades on Hyperliquid based on the lobster's neural firing patterns.

**This is not a demo. This is a real, deployable trading system.**

---

## How It Works

```
Lobster STG Ganglion
      │
  Tungsten Electrodes
      │
  INA333 Bio-Amplifier (50–500× gain)
      │
  Arduino / OpenBCI ADC
      │  (USB Serial)
  SerialReader (Python)
      │
  Bandpass Filter (300–3000 Hz)
  + Notch Filter (60 Hz)
      │
  Spike Detector (MAD threshold)
      │
  Feature Extractor
  (spike rate, ISI, burst score, amplitude)
      │
  Threshold Classifier ─── or ─── ML Classifier (RandomForest)
      │
  Risk Manager
  (daily loss limit, SL/TP, cooldown, position size cap)
      │
  Hyperliquid Exchange API
  (BTC perpetuals, cross margin, configurable leverage)
      │
  Rich Terminal Dashboard
```

### Signal Logic
| Neural State | Features | Signal |
|---|---|---|
| High excitatory firing | Rate ≥ 14 Hz, burst score ≥ 0.65 | **LONG** |
| Low / inhibited | Rate ≤ 5 Hz, burst score ≤ 0.30 | **SHORT** |
| Intermediate | Between thresholds | **HOLD** |

The lobster's **STG** (stomatogastric ganglion) is one of the most studied neural circuits in biology. It contains ~30 neurons and generates complex rhythmic patterns. High-frequency bursting states correlate with active/excited behavioral states; quiescent low-rate firing correlates with inhibited/resting states.

---

## Hardware Requirements

### Option A — Budget Setup (~$50–150)
- **Arduino Uno or Mega** — ADC at 10kHz
- **INA333 instrumentation amplifier breakout** (~$5, Texas Instruments)
- **Tungsten microelectrodes** — 0.001" diameter silver/tungsten wire (~$20)
- **Sylgard 182** — dish coating for pinning tissue (~$30)
- **Petri dish** (100×15mm) — recording chamber
- **Dissecting microscope** — to place electrodes on STG

### Option B — Research Grade (~$500)
- **OpenBCI Cyton** — 8-channel, 250Hz default (configurable higher)
- Proper bio-amp input stage, EMI shielding

### Option C — Maximum Signal Quality (~$1,500+)
- **Intan RHD2132** — designed for neural recording, 30kHz per channel

### Electrode Placement
The **stomatogastric ganglion** sits on the dorsal surface of the lobster stomach. After dissection (see `hardware/arduino/electrode_reader.ino` comments for prep notes), the STG is pinned out in a Sylgard-coated dish with oxygenated lobster saline. Electrodes are placed on the **lateral ventricular nerve (lvn)** or **pyloric dilator nerve (pdn)** for maximal spike yield.

Reference: [JoVE Protocol — Cancer borealis STG dissection](https://app.jove.com/t/1207)

---

## Software Setup

### 1. Clone the repo
```bash
git clone https://github.com/clawduis/The-Trading-Lobster.git
cd The-Trading-Lobster
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure environment
```bash
cp .env.example .env
```

Edit `.env`:
```
HYPERLIQUID_PRIVATE_KEY=0xyour_private_key
HYPERLIQUID_WALLET_ADDRESS=0xYourWalletAddress
USE_TESTNET=true
SERIAL_PORT=/dev/ttyUSB0   # or COM3 on Windows
BAUD_RATE=115200
```

> ⚠️ **Use a dedicated wallet.** Never use your main Hyperliquid wallet. Generate a fresh one and fund it with only the amount you're comfortable trading with.

### 4. Configure trading parameters
Edit `config.yaml` to set:
- `trading.leverage` — default 3x
- `trading.position_size_usd` — default $50 per trade
- `risk.max_daily_loss_pct` — default 5%
- `risk.stop_loss_pct` / `risk.take_profit_pct`
- `classifier.mode` — `"threshold"` or `"ml"`

---

## Running

### Test run (no hardware, no trades)
```bash
python main.py --mock --dry-run
```

### Live signal, no trades
```bash
python main.py --dry-run
```

### Full live system (real signal + real trades)
```bash
python main.py
```

### With testnet (recommended first)
Set `USE_TESTNET=true` in `.env`, then:
```bash
python main.py
```

---

## ML Classifier (Optional)

The default classifier uses hand-tuned thresholds. For better accuracy, train the ML model on your specific lobster's signal characteristics.

### Step 1: Record labeled data
```bash
# Record 2 minutes of high-activity state
python scripts/record_baseline.py --label LONG --duration 120

# Record 2 minutes of low-activity state
python scripts/record_baseline.py --label SHORT --duration 120

# Record 1 minute of intermediate state
python scripts/record_baseline.py --label HOLD --duration 60

# Use mock signal for testing the pipeline
python scripts/record_baseline.py --label LONG --duration 60 --mock
```

### Step 2: Train the model
```bash
python scripts/train_classifier.py
```

### Step 3: Enable ML mode
In `config.yaml`:
```yaml
classifier:
  mode: "ml"
```

---

## Project Structure

```
The-Trading-Lobster/
├── main.py                          # Entry point
├── config.yaml                      # All trading + signal parameters
├── .env.example                     # Environment variable template
├── requirements.txt
│
├── hardware/
│   ├── serial_reader.py             # Reads from Arduino/OpenBCI via serial
│   ├── mock_reader.py               # Synthetic signal generator for testing
│   └── arduino/
│       └── electrode_reader.ino     # Arduino firmware for bio-amp
│
├── processing/
│   ├── filters.py                   # Bandpass + notch filtering
│   ├── spike_detector.py            # MAD-threshold action potential detection
│   └── features.py                  # Spike rate, ISI, burst score extraction
│
├── classifier/
│   ├── threshold.py                 # Rule-based classifier
│   ├── ml_classifier.py             # RandomForest ML classifier
│   └── models/                      # Saved model files (after training)
│
├── trading/
│   ├── hyperliquid_client.py        # Hyperliquid SDK wrapper
│   ├── risk.py                      # Daily loss limit, SL/TP, cooldown
│   └── position_manager.py          # State machine for trade execution
│
├── monitoring/
│   ├── logger.py                    # Structured logging (JSON + console)
│   └── dashboard.py                 # Rich terminal dashboard
│
├── scripts/
│   ├── record_baseline.py           # Record labeled training data
│   └── train_classifier.py          # Train + evaluate ML model
│
└── data/
    └── recordings/                  # CSV recordings (gitignored)
```

---

## Risk Disclosure

This project uses a biological organism's neural activity to make financial decisions. The lobster has no awareness of BTC price action. **Treat all capital deployed through this system as risk capital.** 

Default risk settings:
- Max 5% daily drawdown before automatic halt
- 2.5% stop loss per position
- 5% take profit per position
- 5-minute cooldown between trades
- Hard cap: $500 max position size

Start on testnet. Start small. The lobster does not know what Bitcoin is.

---

## License

MIT

---

*Built with 🦞 and OpenClaw*
