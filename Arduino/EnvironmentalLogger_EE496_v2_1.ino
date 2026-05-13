// ========================================================================================
// EE496 Final Year Project - Environmental Logger
// Student: Luke Buckley
// Student ID: 22504436
//
// Purpose:
//   Environmental data logging for a CO2 control chamber (Arabidopsis research).
//   Measures and logs CO2 concentration, temperature, pressure, humidity, and gas
//   resistance. Data is stored to SD card, displayed on a 20x4 LCD, and transmitted
//   Data is stored to SD card and displayed on a 20x4 LCD.
//
//   NOTE: This is a logging-only unit. The Clippard valve (Pin 44) is held LOW
//   at all times - no CO2 dosing occurs from this unit.
//
// Hardware:
//   - Arduino Mega 2560
//   - K30 CO2 sensor (I2C, address 0x34)
//   - BME680 environmental sensor (I2C) - Temperature, Pressure, Humidity, Gas
//   - RTC PCF8523 (I2C)
//   - 20x4 I2C LCD (address 0x27)
//   - SD card (SPI, CS pin 10)
//   - Clippard valve (Pin 44) - DISABLED
//
// Calibration:
//   CO2 offset of -269 ppm applied. Determined by co-locating this unit with a
//   reference CO2 controller in the same chamber for 5 minutes (08/04/2026).
//   This logger read 869 ppm vs reference reading of 600 ppm.
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
unsigned long intervalco2 = 5000;  // CO2 measurement interval (ms)
const int     chipSelect  = 10;    // SD card chip select pin
const int     Pin44       = 44;    // Clippard valve pin - held LOW in this version
const int     Pin53       = 53;    // SD activity indicator LED

// ========================================================================================
// SENSOR VARIABLES
// ========================================================================================
File dataFile;

// CO2
int    co2Addr = 0x34; // K30 I2C address (7-bit, shifted left)
double co2;            // Current CO2 reading (ppm), calibration offset applied
double pco2;           // Previous CO2 reading (ppm)
float  co2ave5;        // 5-minute CO2 average

// BME680
double temp;           // Temperature (deg C)
double pressure;       // Pressure (mbar)
double humidity;       // Relative humidity (%)
double gas;            // Gas resistance (kOhms)

// Averaging buffers
Average<float> co2_5(60); // 5-minute CO2 average (60 x 5s samples)

// 24-hour CO2 running average - resets at midnight each day
float co2_24h_sum   = 0;
long  co2_24h_count = 0;
float co2_24h_avg   = 0;
int   last_day      = -1; // tracks day for midnight rollover detection


// ========================================================================================
// TIMING VARIABLES
// ========================================================================================
unsigned long previousMillis3  = 0;
unsigned long previousMillis5  = 0;
unsigned long previousMillis7  = 0;
unsigned long previousMillis12 = 0;

// LCD state tracking
int lcd_state  = 0;
int lcd_state2 = 0;
int hr;



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
  // Valve pin - held LOW throughout (logging only)
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
  lcd.setCursor(0, 0); lcd.print("Environmental Logger");
  lcd.setCursor(0, 1); lcd.print("EE 496 FYP");
  lcd.setCursor(0, 2); lcd.print("L.Buckley 22504436");
  lcd.setCursor(0, 3); lcd.print("2026");
  delay(5000);

  pinMode(Pin44, OUTPUT); // Clippard valve - disabled
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

    // Apply calibration offset (-269 ppm)
    co2 = co2 - 269;

    // Update averaging buffer
    co2_5.push(co2);
    co2ave5 = co2_5.mean();

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

    pco2 = co2;
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

  // Screen 1: CO2 reading, 24h average, timestamp
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
    lcd.print("5min= ");
    lcd.print(co2ave5);

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

  // Valve disabled - environmental logging only
  digitalWrite(Pin44, LOW);

  // --- SD card logging (every hour) ---
  unsigned long currentMillis5 = millis();
  if (currentMillis5 - previousMillis5 >= 3600000)
  {
    DateTime now = rtc.now();
    digitalWrite(Pin53, HIGH);
    double t = now.unixtime();

    dataFile.print("EE496 FYP Environmental Log: ");
    dataFile.print("TimeStamp=");      dataFile.print(t);              dataFile.print(", ");
    dataFile.print("Date+Time=");
    dataFile.print(now.year(), DEC);   dataFile.print('/');
    dataFile.print(now.month(), DEC);  dataFile.print('/');
    dataFile.print(now.day(), DEC);    dataFile.print(' ');
    dataFile.print(now.hour(), DEC);   dataFile.print(':');
    dataFile.print(now.minute(), DEC); dataFile.print(':');
    dataFile.print(now.second(), DEC); dataFile.print(", ");
    dataFile.print("CO2=");            dataFile.print(co2);            dataFile.print("ppm, ");
    dataFile.print("CO2_5min=");       dataFile.print(co2ave5);        dataFile.print(", ");
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
