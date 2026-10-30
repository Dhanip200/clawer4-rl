/*
  CLAWER-4 hardware check (Waveshare General Driver for Robots, ESP32).
  READ ONLY: it never moves a servo.

  Every 2 s it prints:
    1. every ST3020 servo found on the bus: ID, position, voltage, temperature, load
    2. the QMI8658 IMU: acceleration (g) and spin (deg/s)
    3. how long one read of all servos takes (the RL brain needs it well under 20 ms)

  Board setting: "ESP32 Dev Module". Serial Monitor: 115200 baud.
  Libraries: SCServo (FT&WS), SensorLib (Lewis He).
*/
#include <SCServo.h>
#include <Wire.h>
#include "SensorQMI8658.hpp"

// Pins on the General Driver for Robots (from Waveshare's ugv_config.h)
#define S_RXD 18   // servo bus
#define S_TXD 19
#define I2C_SDA 32 // IMU
#define I2C_SCL 33

const int MAX_ID = 30;   // look for servo IDs 0..30

SMS_STS st;
SensorQMI8658 imu;
bool imuOk = false;

int foundIds[MAX_ID + 1];
int numFound = 0;

void findServos() {
  numFound = 0;
  for (int id = 0; id <= MAX_ID; id++) {
    if (st.Ping(id) != -1) foundIds[numFound++] = id;
  }
}

void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n=== CLAWER-4 hardware check (read only) ===");

  Serial1.begin(1000000, SERIAL_8N1, S_RXD, S_TXD);
  st.pSerial = &Serial1;
  delay(500);

  // the IMU can sit on either of two addresses
  imuOk = imu.begin(Wire, QMI8658_L_SLAVE_ADDRESS, I2C_SDA, I2C_SCL) ||
          imu.begin(Wire, QMI8658_H_SLAVE_ADDRESS, I2C_SDA, I2C_SCL);
  if (imuOk) {
    imu.configAccelerometer(SensorQMI8658::ACC_RANGE_4G, SensorQMI8658::ACC_ODR_1000Hz);
    imu.configGyroscope(SensorQMI8658::GYR_RANGE_512DPS, SensorQMI8658::GYR_ODR_896_8Hz);
    imu.enableAccelerometer();
    imu.enableGyroscope();
  }
  Serial.println(imuOk ? "IMU: QMI8658 found" : "IMU: NOT found (check I2C)");

  findServos();
  Serial.printf("Servos found: %d\n", numFound);
}

void loop() {
  Serial.println("\n----------------------------------------------");

  // 1. servos
  Serial.println(" ID | position | angle(deg) | volts | temp C | load");
  for (int i = 0; i < numFound; i++) {
    int id = foundIds[i];
    int pos = st.ReadPos(id);
    float volts = st.ReadVoltage(id) / 10.0;
    int temp = st.ReadTemper(id);
    int load = st.ReadLoad(id);
    float deg = (pos - 2047) * 360.0 / 4096.0;   // 2047 = middle
    Serial.printf("%3d | %8d | %10.1f | %5.1f | %6d | %d\n", id, pos, deg, volts, temp, load);
  }
  if (numFound == 0) Serial.println(" (no servos answered: check power, wiring and the servo current switch)");

  // 2. IMU
  if (imuOk) {
    float ax, ay, az, gx, gy, gz;
    imu.getAccelerometer(ax, ay, az);
    imu.getGyroscope(gx, gy, gz);
    Serial.printf("IMU accel g: x %+.2f  y %+.2f  z %+.2f   (lying flat: one axis ~ +-1.00)\n", ax, ay, az);
    Serial.printf("IMU spin deg/s: x %+.1f  y %+.1f  z %+.1f   (still: all ~ 0)\n", gx, gy, gz);
  }

  // 3. timing: how long to read every servo's position + speed once
  unsigned long t0 = micros();
  for (int i = 0; i < numFound; i++) st.FeedBack(foundIds[i]);
  unsigned long us = micros() - t0;
  Serial.printf("Reading %d servos took %.1f ms (budget: 20 ms per brain step)\n", numFound, us / 1000.0);

  delay(2000);
}
