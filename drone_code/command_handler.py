import os
import subprocess
from subprocess import STDOUT, check_output
import time
from datetime import datetime
from images import convert_image
from file_sender import send_file
import math
import zlib
from camera import capture_photo
import csv
import threading
import requests

MAX_HISTORY = 500  # Number of sent packets to retain in memory

# Base command class
class Command:
    name = None  # Should be overridden in subclass

    def execute(self, args, handler):
        raise NotImplementedError("Command must implement execute()")

class StatusCommand(Command):
    name = "STATUS"

    def execute(self, args, handler):
        response = "→ Drone is online and ready"
        handler.send_response(response)
        handler.send_final_token()

class StopCommand(Command):
    name = "STOP"

    def execute(self, args, handler):
        response = "→ Stopping all activity"
        handler.send_response(response)
        handler.send_final_token()

class HelpCommand(Command):
    name = "HELP"

    def execute(self, args, handler):
        # List available commands based on the registered ones
        available = ", ".join(handler.commands.keys())
        response = f"Valid commands: {available}"
        handler.send_response(response)
        handler.send_final_token()


class HistoryCommand(Command):
    name = "HISTORY"

    def execute(self, args, handler):
        try:
            if len(args) == 0:
                return handler.send_response("Usage: HISTORY (# of packets)", handler.rfm9x)
            count = int(args[0])
            history = handler.packet_history
            to_resend = history[-count:] if count <= len(history) else history
            handler.send_response(f"→ Resending last {len(to_resend)} packets", handler.rfm9x)
            for packet in to_resend:
                handler.rfm9x.send(packet)
            handler.send_final_token()
        except Exception as e:
            handler.send_response(f"[REQUEST ERROR] Invalid argument: {e}", handler.rfm9x)
            handler.send_final_token()

class EchoCommand(Command):
    name = "ECHO"

    def execute(self, args, handler):
        try:
            if len(args) == 0:
                handler.send_response("Usage: ECHO (# of packets) (message)", handler.rfm9x)
                handler.send_final_token()
                return

            times = int(args[0])
            message = args[1] if len(args) > 1 else ""

            total_bytes_sent = 0
            start_time = time.time()

            for i in range(times):
                bytes_sent = handler.send_response(message, handler.rfm9x)
                total_bytes_sent += bytes_sent
                time.sleep(0.1)  # simulate delay between packets

            end_time = time.time()
            elapsed_time = end_time - start_time

            # Calculate throughput (bytes per second)
            throughput = total_bytes_sent / elapsed_time if elapsed_time > 0 else 0
            latency_per_packet = elapsed_time / times if times > 0 else 0

            handler.send_response(f"[THROUGHPUT] {throughput:.2f} bytes/sec | [LATENCY] {latency_per_packet:.4f} sec/packet", handler.rfm9x)
            time.sleep(0.1)
            handler.send_final_token()
        except Exception as e:
            handler.send_response(f"[REQUEST ERROR] Invalid argument: {e}", handler.rfm9x)
            time.sleep(0.1)
            handler.send_final_token()


class ConfigCommand(Command):
    name = "CONFIG"

    def execute(self, args, handler):
        try:
            if len(args) == 1 and args[0].upper() == "HELP":
                response = (
                    "CONFIG OPTIONS:\n"
                    "- OUTPUT_LENGTH <32-252>\n"
                    "- LOGGING <true|false>\n"
                    "- TIMESTAMP <true|false>\n"
                    "- CHUNKING <true|false>"
                )
            elif len(args) < 2:
                raise ValueError("Usage: CONFIG <PARAM> <VALUE>")

            else:
                param, value = args[0].upper(), args[1].lower()

                if param == "OUTPUT_LENGTH":
                    new_size = int(value)
                    if 32 <= new_size <= 252:
                        handler.max_packet_size = new_size
                        response = f"Set OUTPUT_LENGTH to {new_size} bytes"
                    else:
                        response = f"Invalid OUTPUT_LENGTH: {new_size} (must be 32-252)"

                elif param == "LOGGING":
                    handler.logging_enabled = value in ["true", "1", "on"]
                    response = f"{'Enabled' if handler.logging_enabled else 'Disabled'} LOGGING"

                elif param == "TIMESTAMP":
                    handler.timestamp_enabled = value in ["true", "1", "on"]
                    response = f"{'Enabled' if handler.timestamp_enabled else 'Disabled'} TIMESTAMP"

                elif param == "CHUNKING":
                    handler.chunking_enabled = value in ["true", "1", "on"]
                    response = f"{'Enabled' if handler.chunking_enabled else 'Disabled'} CHUNKING"

                else:
                    response = f"Unknown CONFIG parameter: {param}"

        except Exception as e:
            response = f"CONFIG error: {e}"

        handler.send_response(response)
        handler.send_final_token()

class ScreenshotCommand(Command):
    name = "SCREENSHOT"
    
    def execute(self, args, handler):
        try:
            # Set image parameters – adjust as needed.
            bit_depth = 4
            size = (128, 128)
            
            # Save the original max packet size or default to 128
            original_max_packet_size = getattr(handler, "max_packet_size", 128)
            max_packet_size = original_max_packet_size

            # If a second argument is provided, try to use it as the new packet size
            if len(args) > 1:
                try:
                    max_packet_size = int(args[1])
                except ValueError:
                    handler.send_response("Invalid packet size. Must be an integer.", handler.rfm9x)
                    handler.send_final_token()
                    return

            # Temporarily override handler.max_packet_size
            handler.max_packet_size = max_packet_size

            # Determine the image path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            image_path = os.path.join(script_dir, "img", args[0])
            
            # Load, dither, and pack image bits (returns a bytearray)
            hex_data = convert_image(image_path, bit_depth=bit_depth, size=size, dithering=False)
            if not hex_data:
                handler.send_response("Image conversion failed", handler.rfm9x)
                return
            
            # Optionally write to terminal log
            with open("terminal.txt", "w") as f:
                f.write(hex_data)
            
            handler.send_response(f"Sending an {size} {bit_depth}bpp image in {max_packet_size}-byte chunks")

            # Send the image data
            if send_file(hex_data, handler):
                # handler.send_response("SCREENSHOT SENT", handler.rfm9x)
                handler.send_final_token()
            else:
                handler.send_response("Failed to send screenshot", handler.rfm9x)
                handler.send_final_token()

            # Restore the original packet size
            handler.max_packet_size = original_max_packet_size
            # handler.send_final_token()
                
        except Exception as e:
            # Restore even if there's an error
            handler.max_packet_size = original_max_packet_size
            handler.send_response(f"[SCREENSHOT ERROR] {e}", handler.rfm9x)
            handler.send_final_token()

class CameraCommand(Command):
    name = "CAMERA"
    
    def execute(self, args, handler):
        try:
            
            import time

            while True:
                image_path = capture_photo()
                if image_path:
                    break  
                print("Retrying capture...")
                time.sleep(1)  

            image_path = capture_photo()
            
            # Set image parameters – adjust as needed.
            bit_depth = 4
            size = (64, 64)
            
            # Load, dither, and pack image bits (returns a bytearray)
            hex_data = convert_image(image_path, bit_depth=bit_depth, size=size, dithering=False)
            if not hex_data:
                handler.send_response("Image conversion failed", handler.rfm9x)
                return
            
            with open("terminal.txt", "w") as f:
                f.write(hex_data)
            
            handler.send_response(f"Sending an {size} {bit_depth}bpp image")
            # # Send the file using file_sender's send_file function
            if send_file(hex_data, handler):
                handler.send_response("SCREENSHOT SENT", handler.rfm9x)
            else:
                handler.send_response("Failed to send screenshot", handler.rfm9x)
                
        except Exception as e:
            handler.send_response(f"[SCREENSHOT ERROR] {e}", handler.rfm9x)

class ResendCommand(Command):
    name = "RESEND"
    
    def execute(self, args, handler):
        """
        Resends specific packets based on a comma-separated list of packet sequence numbers.
        Example command: RESEND 0,2,5
        """
        if not args:
            handler.send_response("Usage: RESEND <packet numbers, comma separated>", handler.rfm9x)
            return
        try:
            # Combine all arguments into one string in case spaces are used.
            indices_str = " ".join(args).replace(":", "").strip()
            # Split by commas (and spaces) to extract packet indices.
            indices = []
            for part in indices_str.split(","):
                for token in part.split():
                    token = token.strip()
                    if token.isdigit():
                        indices.append(int(token))
            if not indices:
                handler.send_response("No valid packet indices provided for RESEND.", handler.rfm9x)
                return
            history = handler.packet_history
            for i in indices:
                try:
                    packet = history[i]
                    handler.rfm9x.send(packet)
                    print(f"Resent packet {i}")
                except IndexError:
                    handler.send_response(f"Packet {i} not found in history.", handler.rfm9x)
        except Exception as e:
            handler.send_response(f"[RESEND ERROR] {e}", handler.rfm9x)

class RunCommand(Command):
    name = "RUN"

    def execute(self, args, handler):
        if not args:
            handler.send_response("Usage: RUN <command> [args...]")
            handler.send_final_token()
            return

        try:
            # Join the args into a full shell command
            shell_cmd = " ".join(args)
            handler.send_response(f"→ Executing: {shell_cmd}")

            # Use Popen to stream output line-by-line
            process = subprocess.Popen(
                shell_cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            # Read lines from stdout as they come
            for line in iter(process.stdout.readline, ''):
                line = line.strip()
                if line:
                    handler.send_response(line)

            process.stdout.close()
            process.wait()

        except subprocess.TimeoutExpired:
            handler.send_response("[ERROR] Command timed out.")
        except Exception as e:
            handler.send_response(f"[ERROR] Failed to execute: {e}")

        handler.send_final_token()


class CommandHandler:
    def __init__(self, rfm9x):
        self.rfm9x = rfm9x
        self.rfm9x.ack_delay = 0.01
        self.rfm9x.node = 1
        self.rfm9x.destination = 2
        self.packet_history = []
        self.max_packet_size = 128
        self.logging_enabled = False
        self.timestamp_enabled = False
        self.chunking_enabled = True
        self.commands = {}
        self.register_commands([
            StatusCommand(),
            StopCommand(),
            HelpCommand(),
            HistoryCommand(),
            EchoCommand(),
            ConfigCommand(),
            ScreenshotCommand(),
            CameraCommand(),
            ResendCommand(), 
            RunCommand(),
        ])


    def register_commands(self, command_list):
        for command in command_list:
            self.commands[command.name] = command


    def send_response(self, response, rfm9x=None):
        from datetime import datetime  # just in case it's not at the top

        rfm9x = rfm9x or self.rfm9x
        encoded_response = response.encode('utf-8')

        # Determine prefix length for timestamp/logging
        prefix_len = 0
        if self.logging_enabled and self.timestamp_enabled:
            prefix_len = 30

        max_data_len = self.max_packet_size - prefix_len

        # Chunk the response
        if self.chunking_enabled:
            chunks = [
                encoded_response[i:i + max_data_len]
                for i in range(0, len(encoded_response), max_data_len)
            ]
        else:
            chunks = [encoded_response]

        total = len(chunks)
        total_bytes_sent = 0  # <--- Track actual bytes sent

        for idx, chunk in enumerate(chunks, start=1):
            if self.logging_enabled:
                if self.timestamp_enabled:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    prefix = f"[{timestamp} {idx}/{total}] "
                else:
                    prefix = f"[{idx}/{total}] "
                payload = prefix.encode('utf-8') + chunk
            else:
                payload = chunk

            print("[DEBUG] Sending payload:", payload)
            rfm9x.send_with_ack(payload)
            self.packet_history.append(payload)

            total_bytes_sent += len(payload)  # <--- Add actual payload length

            if len(self.packet_history) > MAX_HISTORY:
                self.packet_history.pop(0)

        return total_bytes_sent  # <--- Return byte count



    def handle_command(self, command, args):
        try:
            cmd = command.upper()
            if cmd in self.commands:
                self.commands[cmd].execute(args, self)
            else:
                self.send_response(f"[UNIMPLEMENTED COMMAND] {cmd}")
        except Exception as e:
            self.send_response(f"[ERROR] Command handling failed: {e}")

    def send_final_token(self, rfm9x=None):
        rfm9x = rfm9x or self.rfm9x
        FINAL_TOKEN = "END_OF_STREAM"  # Make sure this token does not appear in regular messages.
        final_packet = FINAL_TOKEN.encode('utf-8')
        print("[DEBUG] Sending final token:", final_packet)
        rfm9x.send_with_ack(final_packet)
        self.packet_history.append(final_packet)
        if len(self.packet_history) > MAX_HISTORY:
            self.packet_history.pop(0)