// ========================================================================================
// EE496 Final Year Project - CO2 Control System
// Student: Luke Buckley
// Student ID: 22504436
//
// Purpose:
//   CO2 control system for a plant growth chamber (Arabidopsis research).
//   Measures CO2 concentration, calculates a proportional duty cycle, and opens
//   a Clippard valve (Pin 44) to dose CO2. Logs environmental data to SD card,

//   Data is stored to SD card and displayed on a 20x4 LCD.
//
// Hardware:
//   - Arduino Mega 2560
//   - K30 CO2 sensor (I2C, address 0x34)
//   - BME680 environmental sensor (I2C) - Temperature, Pressure, Humidity, Gas
//   - RTC PCF8523 (I2C)
//   - 20x4 I2C LCD (address 0x27)
//   - SD card (SPI, CS pin 10)
//   - Clippard valve (Pin 44)
//
// Future Forests Project - based on original CO2 Control System v1.1
// ========================================================================================

#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <SPI.h>
#include <SD.h>
#include <Average.h>
#include <RTClib.h>
#include <Adafruit_Sensor.h>
#include <Adafruit_BME680.h>

// ========================================================================================
// DEFINES
// ========================================================================================
#define SEALEVELPRESSURE_HPA (1013.25)
#define BACKLIGHT_PIN        (3)
#define LED_ADDR             (0x27)    // Try 0x3F if LCD doesn't respond

// ========================================================================================
// HARDWARE OBJECTS
// ========================================================================================
LiquidCrystal_I2C lcd(LED_ADDR, 20, 4);
RTC_PCF8523 rtc;
Adafruit_BME680 bme;

// ========================================================================================
// CONFIGURATION - SET THESE BEFORE UPLOADING
// ========================================================================================
unsigned long intervalco2  = 5000;  // CO2 measurement interval (ms)
const int     chipSelect   = 10;    // SD card chip select pin
const double  treatment    = 1100;  // CO2 target setpoint (ppm)
const int     Pin44        = 44;    // Clippard valve pin
const int     Pin53        = 53;    // SD activity indicator LED

// ========================================================================================
// SENSOR VARIABLES
// ========================================================================================
File dataFile;

// CO2
int    co2Addr = 0x34; // K30 I2C address (7-bit, shifted left)
double co2;            // Current CO2 reading (ppm)
double pco2;           // Previous CO2 reading (ppm)
float  co2ave1;        // 1-minute CO2 average (used for valve control logic)
float  co2ave5;        // 5-minute CO2 average

// BME680
double temp;           // Temperature (deg C)
double pressure;       // Pressure (mbar)
double humidity;       // Relative humidity (%)
double gas;            // Gas resistance (kOhms)

// Averaging buffers
Average<float> co2_5(60);  // 5-minute CO2 average (60 x 5s samples)
Average<float> co2_1(12);  // 1-minute CO2 average (12 x 5s samples)
Average<float> duty_5(60); // 5-minute duty cycle average

// 24-hour CO2 running average - resets at midnight each day
float co2_24h_sum   = 0;
long  co2_24h_count = 0;
float co2_24h_avg   = 0;
int   last_day      = -1; // tracks day for midnight rollover detection

// ========================================================================================
// VALVE CONTROL VARIABLES
// ========================================================================================
double duty0    = 300;  // Initial duty cycle starting point
double duty1;           // Calculated valve duty cycle
double dutyp;           // Previous duty cycle
float  dutyave5;        // 5-minute duty cycle average
double interval1 = 500; // Valve PWM period (ms)
int    State1    = LOW; // Valve state
int    hr;              // Current hour (for time-based valve shutoff)

// ========================================================================================
// TIMING VARIABLES
// ========================================================================================
unsigned long previousMillis1  = 0;
unsigned long previousMillis3  = 0;
unsigned long previousMillis5  = 0;
unsigned long previousMillis7  = 0;
unsigned long previousMillis12 = 0;

// LCD state tracking
int lcd_state  = 0;
int lcd_state2 = 0;



// ========================================================================================
// SOFTWARE RESET
// Used as watchdog for sensor dropout recovery
// ========================================================================================
void software_Reset()
{
  asm volatile ("  jmp 0");
}

// ========================================================================================
// CO2 SENSOR READ FUNCTION (K30 via I2C)
// Returns CO2 value in ppm on success, 0 on checksum failure
// ========================================================================================
int readCO2()
{
  int co2_value = 0;

  Wire.beginTransmission(co2Addr);
  Wire.write(0x22);
  Wire.write(0x00);
  Wire.write(0x08);
  Wire.write(0x2A);
  Wire.endTransmission();

  delay(10); // Allow sensor time to process

  Wire.requestFrom(co2Addr, 4);
  byte i = 0;
  byte buffer[4] = {0, 0, 0, 0};
  while (Wire.available())
  {
    buffer[i] = Wire.read();
    i++;
  }

  co2_value  = 0;
  co2_value |= buffer[1] & 0xFF;
  co2_value  = co2_value << 8;
  co2_value |= buffer[2] & 0xFF;

  byte sum = buffer[0] + buffer[1] + buffer[2];
  if (sum == buffer[3])
    return co2_value;
  else
    return 0;
}

// ========================================================================================
// SETUP
// ========================================================================================
void setup()
{
  pinMode(Pin44, OUTPUT);
  digitalWrite(Pin44, LOW);

  Serial.begin(9600);

  Wire.begin();
  lcd.init();
  lcd.backlight();

  // RTC - update DateTime before each upload, no leading zeros on month/day
  digitalWrite(10, HIGH);
  rtc.adjust(DateTime(2026, 4, 29, 12, 0, 0)); // <-- UPDATE BEFORE UPLOADING
  pinMode(SS, OUTPUT);

  // SD card
  if (!SD.begin(chipSelect))
  {
    lcd.clear();
    lcd.setCursor(0, 1);
    lcd.print("Card Fail");
    while (1);
  }
  lcd.clear();
  lcd.setCursor(0, 1);
  lcd.print("CARD PASS");

  dataFile = SD.open("datalog.txt", FILE_WRITE);
  if (!dataFile)
  {
    lcd.setCursor(0, 2);
    lcd.print("error datalog");
    while (1);
  }
  lcd.setCursor(0, 2);
  lcd.print("datalog cool");

  // BME680
  if (!bme.begin())
  {
    lcd.setCursor(0, 3);
    lcd.print("BME680 fail");
  }
  else
  {
    lcd.setCursor(0, 3);
    lcd.print("BME680 pass");
  }
  delay(1000);

  pinMode(BACKLIGHT_PIN, OUTPUT);
  digitalWrite(BACKLIGHT_PIN, HIGH);

  // Startup screen
  lcd.home();
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("CO2 Control System");
  lcd.setCursor(0, 1); lcd.print("EE 496 FYP");
  lcd.setCursor(0, 2); lcd.print("L.Buckley 22504436");
  lcd.setCursor(0, 3); lcd.print("2026");
  delay(5000);

  pinMode(Pin44, OUTPUT); // Clippard valve
  pinMode(Pin53, OUTPUT); // SD activity LED

  // BME680 oversampling and filter settings
  bme.setTemperatureOversampling(BME680_OS_8X);
  bme.setHumidityOversampling(BME680_OS_2X);
  bme.setPressureOversampling(BME680_OS_4X);
  bme.setIIRFilterSize(BME680_FILTER_SIZE_3);
  bme.setGasHeater(320, 150); // 320 deg C for 150ms

  return;
}

// ========================================================================================
// MAIN LOOP
// ========================================================================================
void loop()
{
  // --- CO2 measurement (every 5 seconds) ---
  unsigned long currentMillis3 = millis();
  if (currentMillis3 - previousMillis3 >= intervalco2)
  {
    // Attempt CO2 read up to 3 times; fall back to previous value if all fail
    co2 = readCO2();
    if (co2 <= 0) co2 = readCO2();
    if (co2 <= 0) co2 = readCO2();
    if (co2 <= 0) co2 = pco2;

    // Update averaging buffers
    co2_1.push(co2);
    co2_5.push(co2);
    duty_5.push(duty1);

    co2ave1  = co2_1.mean();
    co2ave5  = co2_5.mean();
    dutyave5 = duty_5.mean();

    // Sensor dropout watchdog - reset if 1-min average is bad and duty is high
    if (co2ave1 <= 0 && duty1 >= 425)
      software_Reset();

    // Proportional CO2 control using 15-second projection
    float p10co2   = co2_1.get(3);                       // CO2 reading ~15 seconds ago
    float m        = (co2 - p10co2) / 3;                 // Rate of change
    float proj_co2 = co2 + 3 * m;                        // Projected CO2 in ~15 seconds
    float proj_dev = (treatment - proj_co2) / treatment; // Deviation from target

    duty1 = duty0 + (5 * proj_dev);   // Normal correction
    if (co2ave1 > 1120)
      duty1 = duty0 + (20 * proj_dev); // Stronger correction above 1120ppm

    if (duty1 >= 500) duty1 = 500;
    if (duty1 <= 0)   duty1 = duty0;

    // 24-hour running average - resets at midnight
    DateTime nowCO2 = rtc.now();
    if (last_day == -1) last_day = nowCO2.day();
    if (nowCO2.day() != last_day)
    {
      co2_24h_avg   = (co2_24h_count > 0) ? (co2_24h_sum / co2_24h_count) : 0;
      co2_24h_sum   = 0;
      co2_24h_count = 0;
      last_day      = nowCO2.day();
    }
    if (co2 > 0)
    {
      co2_24h_sum  += co2;
      co2_24h_count++;
      co2_24h_avg   = co2_24h_sum / co2_24h_count;
    }

    pco2  = co2;
    dutyp = duty0;
    duty0 = duty1;
    previousMillis3 = currentMillis3;
  }



  // --- BME680 reading (every 10 seconds) ---
  unsigned long currentMillis7 = millis();
  if (currentMillis7 - previousMillis7 >= 10000)
  {
    if (!bme.performReading())
    {
      temp     = -7999;
      pressure = -7999;
      humidity = -7999;
      gas      = -7999;
    }
    else
    {
      temp     = bme.temperature;
      pressure = bme.pressure / 100.0;
      humidity = bme.humidity;
      gas      = bme.gas_resistance / 1000.0;
    }
    previousMillis7 = currentMillis7;
  }

  // --- LCD display (alternates between two screens every 5 seconds) ---
  unsigned long currentMillis12 = millis();
  if (currentMillis12 - previousMillis12 <= 5000)  lcd_state = 0;
  if (currentMillis12 - previousMillis12 > 5000)   lcd_state = 1;
  if (currentMillis12 - previousMillis12 >= 10000)
  {
    lcd_state2 = 0;
    lcd_state  = 0;
    previousMillis12 = currentMillis12;
  }

  // Screen 1: CO2 reading, 24h average, duty cycle, timestamp
  if (lcd_state == 0 && lcd_state2 == 0)
  {
    lcd.clear();
    DateTime now = rtc.now();
    hr = now.hour();

    lcd.setCursor(0, 0);
    lcd.print("CO2 ppm = ");
    lcd.print(co2);

    lcd.setCursor(0, 1);
    lcd.print("24h avg= ");
    lcd.print(co2_24h_avg);

    lcd.setCursor(0, 2);
    lcd.print("duty = ");
    lcd.print(duty1);
    lcd.print(" ");
    lcd.print(dutyp);

    lcd.setCursor(0, 3);
    lcd.print(now.year(), DEC);   lcd.print('/');
    lcd.print(now.day(), DEC);    lcd.print('/');
    lcd.print(now.month(), DEC);  lcd.print(' ');
    lcd.print(now.hour(), DEC);   lcd.print(':');
    lcd.print(now.minute(), DEC); lcd.print(':');
    lcd.print(now.second(), DEC);

    lcd_state2 = 1;
  }

  // Screen 2: BME680 environmental readings
  if (lcd_state == 1 && lcd_state2 == 1)
  {
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("temp = ");     lcd.print(temp);
    lcd.setCursor(0, 1); lcd.print("press = ");    lcd.print(pressure);
    lcd.setCursor(0, 2); lcd.print("humidity = "); lcd.print(humidity);
    lcd.setCursor(0, 3); lcd.print("gas = ");      lcd.print(gas);
    lcd_state2 = 2;
  }

  // --- Valve PWM control ---
  unsigned long currentMillis1 = millis();
  unsigned long startT1 = interval1 - duty1;

  if (currentMillis1 - previousMillis1 <= startT1)
    State1 = LOW;
  else
  {
    State1 = HIGH;
    if (duty1 >= 499)   State1 = HIGH;
    if (hr > 19)        State1 = LOW;  // Valve closed overnight
    if (hr < 6)         State1 = LOW;
    if (co2ave1 > 1800) State1 = LOW;  // Safety ceiling - 1800ppm hard cutoff
  }

  if (currentMillis1 - previousMillis1 > interval1)
    previousMillis1 = currentMillis1;

  digitalWrite(Pin44, State1);

  // --- SD card logging (every hour) ---
  unsigned long currentMillis5 = millis();
  if (currentMillis5 - previousMillis5 >= 3600000)
  {
    DateTime now = rtc.now();
    digitalWrite(Pin53, HIGH);
    double t = now.unixtime();

    dataFile.print("EE496 FYP CO2 Control Log: ");
    dataFile.print("TimeStamp=");      dataFile.print(t);              dataFile.print(", ");
    dataFile.print("Date+Time=");
    dataFile.print(now.year(), DEC);   dataFile.print('/');
    dataFile.print(now.month(), DEC);  dataFile.print('/');
    dataFile.print(now.day(), DEC);    dataFile.print(' ');
    dataFile.print(now.hour(), DEC);   dataFile.print(':');
    dataFile.print(now.minute(), DEC); dataFile.print(':');
    dataFile.print(now.second(), DEC); dataFile.print(", ");
    dataFile.print("CO2=");            dataFile.print(co2);            dataFile.print("ppm, ");
    dataFile.print("Duty=");           dataFile.print(duty1);          dataFile.print(", ");
    dataFile.print("CO2_5min=");       dataFile.print(co2ave5);        dataFile.print(", ");
    dataFile.print("Duty_5min=");      dataFile.print(dutyave5);       dataFile.print(", ");
    dataFile.print("BME680: ");
    dataFile.print("Temperature=");    dataFile.print(temp);           dataFile.print("C, ");
    dataFile.print("Pressure=");       dataFile.print(pressure);       dataFile.print("mbar, ");
    dataFile.print("Humidity=");       dataFile.print(humidity);       dataFile.print("%, ");
    dataFile.print("Gas=");            dataFile.print(gas);            dataFile.print("kOhms, ");
    dataFile.print("CO2_24h_avg=");    dataFile.println(co2_24h_avg);

    dataFile.flush();
    previousMillis5 = currentMillis5;
  }
}
