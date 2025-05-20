import { SerialPort } from 'serialport';
import { WebSocketServer } from 'ws';

const devicePath = '/dev/tty.usbserial-0001'

const port = new SerialPort({ path: devicePath, baudRate: 115200 });
const wss = new WebSocketServer({ port: 8081 });

let latestData = '';

port.on('data', (data) => {  
  latestData += data.toString();
  if (latestData.includes('\n')) {
    const [line, ...rest] = latestData.split('\n');    
    latestData = rest.join('\n');
    const match = line.match(/(-?\d+\.?\d*)\t\s*(-?\d+\.?\d*)\t\s*(-?\d+\.?\d*)/);
    if (match) {
      const [, roll, pitch, yaw] = match.map(Number);
      wss.clients.forEach(ws => {
        ws.send(JSON.stringify({ roll, pitch, yaw }));
      });
    }
  }
});
