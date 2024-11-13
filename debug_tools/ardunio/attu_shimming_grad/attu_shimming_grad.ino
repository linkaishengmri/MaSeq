
#include "EEPROM.h"


const int CLK_PIN = 4;
const int LE_PIN = 2;
const int SI_PIN = 3;
int init_start = 1;

const int clkPin[2] = {10,9};       // clock
const int enablePin[2] = {11,8}; 
 

const int sdaPin[8] = {12,14,15,16,7,6,5,17};


void volt_write_single_ch(uint32_t sd, uint8_t volt_channel, int save_to_rom = 1) {

  if(volt_channel >=8 || volt_channel < 0) return;

  if(save_to_rom) 
  {
    uint8_t Ebyte1=(sd>>16) & (0xFF);
    uint8_t Ebyte2=(sd>>8) & (0xFF);
    uint8_t Ebyte3=(sd>>0) & (0xFF);
    
    EEPROM.write((volt_channel+1)*3, (sd>>16) & (0xFF));
    EEPROM.write((volt_channel+1)*3+1, (sd>>8) & (0xFF));
    EEPROM.write((volt_channel+1)*3+2, (sd>>0) & (0xFF));
    //write24bitToEEPROM(3, sd);
  }
  int board = volt_channel / 4;
  int clk = clkPin[board];
  int enable = enablePin[board];
  int data_pin = sdaPin[volt_channel];

  digitalWrite(enable, LOW);

  for (int i = 23; i >= 0; i--) {
    uint8_t bitData = (sd >> (i)) & 0x01;   
    
    digitalWrite(data_pin, bitData);

    digitalWrite(clk, HIGH);
    delayMicroseconds(10);  

    digitalWrite(clk, LOW);
    delayMicroseconds(10);   
  }

  digitalWrite(enable, HIGH);

}
  

void attu_write(byte data, int attu_channel, int save_to_rom = 1) {
  int CLK, LE, SI;
  CLK = CLK_PIN;
  LE = LE_PIN;
  SI = SI_PIN;

  byte addr;
  if(attu_channel == 0){
    addr = 0x00;
    if(save_to_rom) EEPROM.write(0, data);
  } else if (attu_channel == 1){
    addr = 0x07;
    if(save_to_rom) EEPROM.write(1, data);
  } else {
    return;
  }

  // Set LE low initially
  digitalWrite(LE, LOW);

  //Send 16 clock cycles
  for (int i = 0; i < 16; i++) {
    // Determine the current bit to send (low bit first)
    byte bitToSend = (i < 8) ? (data>> i) & 0x01 : (addr >> (i - 8)) & 0x01;

    // Set the SI pin to the current bit
    digitalWrite(SI, bitToSend);

    // Generate clock pulse
    digitalWrite(CLK, HIGH);
    delayMicroseconds(10); // CLK high for 10μs
    digitalWrite(CLK, LOW);
    delayMicroseconds(10); // CLK low for 10μs
  }

  // Send LE pulse
  digitalWrite(LE, HIGH); // Set LE high
  delayMicroseconds(10);       // LE high for 10μs
  digitalWrite(LE, LOW);  // Set LE low
}

void handleAttenuation(int ch) {

    uint8_t attuValue = Serial.read(); // xx
    uint8_t checkValue = Serial.read();  

    if ((checkValue & 0xF0)>>4 == ch && (checkValue & 0x0F) == 0x0E) {
      attu_write(attuValue, ch);
    } else {
      return;
    }
}

void handleVoltage(int ch) {
  uint64_t voltByte1 = Serial.read(); // xx
  uint64_t voltByte2 = Serial.read(); // xx
  uint64_t voltByte3 = Serial.read(); // xx
  uint8_t checksum = Serial.read(); // 
  uint32_t datasend = ((((voltByte1<<16) | (voltByte2<<8) | (voltByte3<<0)) << 2) |  0x100000) & 0x1FFFFF;

    if (checksum == ((ch << 4) | (0x0F))) { 
        volt_write_single_ch(datasend, ch, 1);
    } else {
        
    }
}
void setup() {
  
  Serial.begin(9600);
  // Initialize pins as output
  pinMode(CLK_PIN, OUTPUT);
  pinMode(LE_PIN, OUTPUT);
  pinMode(SI_PIN, OUTPUT);
  pinMode(10,OUTPUT);
  for(int x=0;x<2;x++)
  {
    pinMode(clkPin[x], OUTPUT);

    pinMode(enablePin[x], OUTPUT);
  }
  for(int x=0;x<8;x++)
  {
    pinMode(sdaPin[x], OUTPUT);
  }
  init_start = 1;
   //digitalWrite(12, HIGH);
}

void loop() {
  delay(50);
  if(init_start)
  { 
    digitalWrite(enablePin[0], HIGH);
    digitalWrite(enablePin[1], HIGH);
    delay(500);
    attu_write(EEPROM.read(0), 0, 0);
    attu_write(EEPROM.read(1), 1, 0);
    // TODO: write init volt
    volt_write_single_ch(0x400004, 0, 0);  delay(1);
    volt_write_single_ch(0x400004, 1, 0);  delay(1);
    volt_write_single_ch(0x400004, 2, 0);  delay(1);
    volt_write_single_ch(0x400004, 3, 0);  delay(1);
    volt_write_single_ch(0x400004, 4, 0);  delay(1);
    volt_write_single_ch(0x400004, 5, 0);  delay(1);
    volt_write_single_ch(0x400004, 6, 0);  delay(1);
    volt_write_single_ch(0x400004, 7, 0);  delay(1);


    volt_write_single_ch(0x200002, 0, 0);  delay(1);
    volt_write_single_ch(0x200002, 1, 0);  delay(1);
    volt_write_single_ch(0x200002, 2, 0);  delay(1);
    volt_write_single_ch(0x200002, 3, 0);  delay(1);
    volt_write_single_ch(0x200002, 4, 0);  delay(1);
    volt_write_single_ch(0x200002, 5, 0);  delay(1);
    volt_write_single_ch(0x200002, 6, 0);  delay(1);
    volt_write_single_ch(0x200002, 7, 0);  delay(1);


    // EEPROM.write(3,  (0x01));
    // EEPROM.write(4,  (0xFF));
    // EEPROM.write(5,  (0xFF));
    // int i=0;
    //   uint64_t Ebyte1 = EEPROM.read(3) & 0xFF;
    //   uint64_t Ebyte2 = EEPROM.read(4) & 0xFF;
    //   uint64_t Ebyte3 = EEPROM.read(5) & 0xFF;
      

    // uint32_t datasend = (((Ebyte1<<16) | (Ebyte2<<8) | (Ebyte3<<0)) ) ;
    //   volt_write_single_ch(datasend, i, 0);  delay(1);
    
    for(int i=0;i<8;i++)
    {
      uint64_t Ebyte1 = EEPROM.read((i+1)*3);
      uint64_t Ebyte2 = EEPROM.read((i+1)*3+1);
      uint64_t Ebyte3 = EEPROM.read((i+1)*3+2);
      uint32_t datasend = (((Ebyte1<<16) | (Ebyte2<<8) | (Ebyte3<<0)) );
      volt_write_single_ch(datasend, i, 0);  delay(1);
    }
    init_start = 0;
  }

  // Serial.print("1");
  // return;
  
  if (1 && Serial.available() > 0) {
        uint8_t command = Serial.read();

        switch (command) {
            case 0xFE: 
                Serial.write(0xFD); 
                break;

            case 0xE0: 
                handleAttenuation(0);
                break;

            case 0xE1: 
                handleAttenuation(1);
                break;

            case 0xF0:  
                handleVoltage(0);
                break;

            case 0xF1:  
                handleVoltage(1);
                break;

            case 0xF2:  
                handleVoltage(2);
                break;

            case 0xF3:  
                handleVoltage(3);
                break;

            case 0xF4:  
                handleVoltage(4);
                break;

            case 0xF5:  
                handleVoltage(5);
                break;

            case 0xF6:  
                handleVoltage(6);
                break;

            case 0xF7:  
                handleVoltage(7);
                break;

            default:
                break;
        }
        
    }

}  





