import board
import busio
import digitalio
import adafruit_rfm9x

RADIO_FREQ_MHZ = 915.0
BAUD_RATE = 19200

def get_lora_radio():
    spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
    chip_select = digitalio.DigitalInOut(board.CE1)
    reset = digitalio.DigitalInOut(board.D25)

    rfm9x = adafruit_rfm9x.RFM9x(spi, chip_select, reset, RADIO_FREQ_MHZ, baudrate=BAUD_RATE)
    rfm9x.tx_power = 23
    rfm9x.signal_bandwidth = 500000
    rfm9x.spreading_factor = 7
    rfm9x.coding_rate = 5
    rfm9x.enable_crc = True
    rfm9x.preamble_length = 6

    print("LoRa transceiver is initialized and tuned.")
    return rfm9x
