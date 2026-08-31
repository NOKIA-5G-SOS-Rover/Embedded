// Bridge: relays movement commands from the backend's dashboardHub to the Arduino's
// serial port. Run on the companion computer the Arduino is plugged into.
//
// Env vars:
//   BACKEND_URL   backend base URL reachable from this host, e.g. http://92.87.91.146:5000
//   ROVER_ID      must match the id the backend broadcasts to; default ROVER-1
//   SERIAL_PORT   default /dev/ttyACM0  (use /dev/ttyUSB0 for a CH340-based clone)
//   BAUD_RATE     default 115200 (must match Serial.begin in sketch.ino)

import { WebSocket as Ws } from 'ws';
// @microsoft/signalr needs a global WebSocket (native on Node 21+; polyfill below for older).
if (!globalThis.WebSocket) globalThis.WebSocket = Ws;

import * as signalR from '@microsoft/signalr';
import { SerialPort } from 'serialport';

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:5000';
const ROVER_ID = process.env.ROVER_ID || 'ROVER-1';
const SERIAL_PORT_PATH = process.env.SERIAL_PORT || '/dev/ttyACM0';
const BAUD_RATE = Number(process.env.BAUD_RATE || 115200);

// MUST match the switch(cmd) in sketch/sketch.ino (f b s l r q e). Editing one side
// without the other silently breaks the rover.
const COMMAND_TO_CHAR = {
  forward: 'f',
  backward: 'b',
  stop: 's',
  'turn-left': 'l',
  'turn-right': 'r',
  'arc-left': 'q',
  'arc-right': 'e',
  'reverse-arc-left': 'z',
  'reverse-arc-right': 'c',
};

let port = null;

function initSerial() {
  port = new SerialPort({ path: SERIAL_PORT_PATH, baudRate: BAUD_RATE, autoOpen: false });
  port.on('open', () => console.log(`[serial] opened ${SERIAL_PORT_PATH}@${BAUD_RATE}`));
  port.on('error', (err) => console.error('[serial] error:', err.message));
  port.on('close', () => console.warn('[serial] closed; will reopen on next command'));
}

function ensureOpen(then) {
  if (port && port.isOpen) { then(); return; }
  if (!port) initSerial();
  port.open((err) => {
    if (err) { console.error('[serial] open failed:', err.message); return; }
    // The Uno resets when the port opens (DTR toggle); its bootloader runs ~1.5s
    // before sketch.ino is ready to read. Held open for the process lifetime, so
    // this delay only bites once (or after a drop/reopen).
    setTimeout(then, 1500);
  });
}

function sendChar(char) {
  ensureOpen(() => {
    port.write(char + '\n', (err) => {
      if (err) console.error('[serial] write failed:', err.message);
      else console.log(`[serial] wrote '${char}'`);
    });
  });
}

const connection = new signalR.HubConnectionBuilder()
  .withUrl(`${BACKEND_URL}/dashboardHub`)
  .withAutomaticReconnect()
  .build();

connection.on('ReceiveCommand', (dto) => {
  const word = dto && dto.command;
  const char = COMMAND_TO_CHAR[word];
  if (!char) { console.log(`[bridge] ignoring unmapped command '${word}'`); return; }
  console.log(`[bridge] ${word} -> ${char}`);
  sendChar(char);
});

async function register() {
  await connection.invoke('RegisterRobot', ROVER_ID);
  console.log(`[bridge] registered as ${ROVER_ID} — listening for commands`);
}

connection.onreconnecting(() => console.warn('[bridge] reconnecting...'));
connection.onreconnected(() => {
  console.log('[bridge] reconnected; re-registering');
  register().catch((e) => console.error('[bridge] re-register failed:', e.message));
});
connection.onclose(() => console.error('[bridge] connection closed'));

async function main() {
  initSerial();
  port.open((err) => {
    if (err) console.error('[serial] initial open failed:', err.message, '(will retry on first command)');
  });

  await connection.start();
  console.log(`[bridge] connected to ${BACKEND_URL}/dashboardHub`);
  await register();
}

main().catch((err) => {
  console.error('[bridge] fatal:', err);
  process.exit(1);
});
