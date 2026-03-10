/*
 * electrode_reader.ino — The Trading Lobster
 * ──────────────────────────────────────────
 * Arduino firmware for bio-signal acquisition.
 *
 * Hardware setup:
 *   - INA333 (or INA128) instrumentation amplifier
 *     - IN+ → electrode (silver/tungsten wire on STG ganglion)
 *     - IN- → reference electrode (in same saline bath)
 *     - REF → Arduino GND
 *     - VS+ → Arduino 5V, VS- → GND
 *     - Gain resistor (Rg): 10kΩ → gain ≈ 50× (adjust for signal level)
 *   - INA333 VOUT → Arduino A0
 *
 * Output protocol:
 *   - Sends one signed 16-bit ADC value per line over Serial
 *   - Format: "-1024\n", "2048\n", etc.
 *   - Sampling rate: controlled by SAMPLE_RATE_HZ
 *
 * Compatible with Python SerialReader (hardware/serial_reader.py)
 *
 * Wiring:
 *   A0 ← INA333 Vout (electrode differential signal)
 *   GND → common ground for INA333 and saline reference
 */

#define ANALOG_PIN      A0
#define SAMPLE_RATE_HZ  10000        // 10kHz (safe for Arduino Uno ADC)
#define ADC_MIDPOINT    512          // Center of 10-bit ADC (0-1023)
#define SERIAL_BAUD     115200

// Microseconds per sample
const unsigned long SAMPLE_INTERVAL_US = 1000000UL / SAMPLE_RATE_HZ;

unsigned long last_sample_us = 0;

void setup() {
  Serial.begin(SERIAL_BAUD);
  analogReference(DEFAULT);         // Use 5V reference
  
  // Wait for serial connection
  while (!Serial) { delay(1); }
  
  // Send init message (Python reader ignores non-numeric lines)
  Serial.println("LOBSTER_READY");
  
  last_sample_us = micros();
}

void loop() {
  unsigned long now = micros();
  
  if (now - last_sample_us >= SAMPLE_INTERVAL_US) {
    last_sample_us = now;
    
    // Read ADC and center around zero
    int raw = analogRead(ANALOG_PIN) - ADC_MIDPOINT;
    
    // Send as signed integer, newline-terminated
    Serial.println(raw);
  }
}

/*
 * CALIBRATION NOTES
 * ─────────────────
 * 1. Before connecting electrodes, short IN+ to IN- and verify ~0 output
 * 2. Neural spikes should be 50–500µV at the electrode tip
 * 3. With gain=50, a 200µV spike → 10mV at VOUT → ~2 ADC units (at 5V ref)
 *    Increase gain (decrease Rg) if spikes are too small
 *    Rg = 100kΩ/(gain-1): gain=100 → Rg=1kΩ, gain=500 → Rg=200Ω
 * 4. Use shielded twisted-pair cable from electrode to INA333
 * 5. Keep electrode wires short (<30cm) to minimize noise pickup
 *
 * UPGRADING TO HIGHER SAMPLE RATE
 * ─────────────────────────────────
 * Arduino Uno ADC is limited to ~10kHz at full precision.
 * For 30kHz recordings (needed for full AP waveforms), use:
 *   - Arduino Due (12-bit ADC, up to 1MHz)
 *   - Teensy 4.0 (12-bit, up to 500kHz)
 *   - Dedicated ADC: ADS1115, MCP3208, or AD7768
 *
 * For research-grade recordings, consider OpenBCI Cyton board or
 * Intan RHD2132 — both designed specifically for neural recording.
 */
