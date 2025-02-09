
//phone call 


import Fastify from 'fastify';
import WebSocket from 'ws';
import dotenv from 'dotenv';
import fastifyFormBody from '@fastify/formbody';
import fastifyWs from '@fastify/websocket';
import twilio from 'twilio';

dotenv.config();

// Retrieve the OpenAI API key from environment variables. You must have OpenAI Realtime API access.
const { OPENAI_API_KEY } = process.env;
if (!OPENAI_API_KEY) {
    console.error('Missing OpenAI API key. Please set it in the .env file.');
    process.exit(1);
}

// Initialize Fastify
const fastify = Fastify();
fastify.register(fastifyFormBody);
fastify.register(fastifyWs);


// Constants
const TWILIO_PHONE = process.env.TWILIO_PHONE;
const VOICE = 'alloy';
const PORT = process.env.PORT || 3000; // Allow dynamic port assignment
// List of Event Types to log to the console. See OpenAI Realtime API Documentation. (session.updated is handled separately.)
const LOG_EVENT_TYPES = [
    'response.content.done',
    'rate_limits.updated',
    'response.done',
    'input_audio_buffer.committed',
    'input_audio_buffer.speech_stopped',
    'input_audio_buffer.speech_started',
    'session.created'
];

// Track active calls and their instructions
const activeCallInstructions = new Map();
const callStatuses = new Map();
const callInformation = new Map();

// Root Route
fastify.get('/', async (request, reply) => {
  reply.send({ message: 'Twilio Media Stream Server is running!' });
});

// Create a call to the user
// takes two arguments: userPhone (user's phone number) and instruction (the prompt for the AI)
async function createCall(userPhone, instruction) {
  // Validate input
  if (!userPhone || !instruction) {
    throw new Error('Missing required arguments: userPhone and instruction are required');
  }

  const twilioClient = twilio(process.env.TWILIO_ACCOUNT_SID, process.env.TWILIO_AUTH_TOKEN);
  const callDT = new Date().toISOString(); // used to identify call through all function passes
  activeCallInstructions.set(callDT, instruction);
  callInformation.set(callDT, { dateTime: callDT, transcription: '' });


  const call = await twilioClient.calls.create({
    to: userPhone,
    from: TWILIO_PHONE,
    url: `${process.env.SERVER_URL}/api/incoming-call?callDT=${callDT}`,
    record: true,
    statusCallback: process.env.SERVER_URL + "/api/status-callback",
    statusCallbackEvent: ['initiated', 'ringing', 'answered', 'completed', 'failed', 'busy', 'no-answer'],
    statusCallbackMethod: 'POST'
  });

  callStatuses.set(call.sid, 'initiated');

  // Return a promise that resolves when the call is completed
  return new Promise((resolve) => {
    const checkCallStatus = setInterval(() => {
      const status = callStatuses.get(call.sid);
      if (status === 'completed') {
        clearInterval(checkCallStatus);
        const callInfo = callInformation.get(callDT);
        resolve(callInfo); // Resolve with the call information
      }
    }, 1000); // Check every second
  });
}


// Route for Twilio to handle incoming and outgoing calls
// <Say> punctuation to improve text-to-speech translation
fastify.all('/api/incoming-call', async (request, reply) => {
  const callDT = request.query.callDT;
  const twimlResponse = `<?xml version="1.0" encoding="UTF-8"?>
    <Response>
        <Say>You are now connected to Open-A.I.</Say>
        <Connect>
            <Stream url="wss://${request.headers.host}/api/media-stream/${callDT}" />
        </Connect>
    </Response>`;
  reply.type('text/xml').send(twimlResponse);
});


// WebSocket route for media-stream
fastify.register(async (fastify) => {
  fastify.get('/api/media-stream/:callDT', { websocket: true }, (connection, req) => {
      console.log('Client connected');

      // Extract callDT from URL path and get the instruction for the call
      const urlPath = req.url;
      const callDT = urlPath.split('media-stream/')[1]?.split('?')[0];
      const instruction = activeCallInstructions.get(callDT);

      // Establish websocket connection to OpenAI Realtime API
      const openAiWs = new WebSocket('wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17', {//2024-10-01 previous option
          headers: {
              Authorization: `Bearer ${OPENAI_API_KEY}`,
              "OpenAI-Beta": "realtime=v1"
          }
      });

      let streamSid = null;
      const sendSessionUpdate = () => {
          const sessionUpdate = {
              type: 'session.update',
              session: {
                  turn_detection: { type: 'server_vad' }, // Use server-based VAD for turn detection
                  input_audio_format: 'g711_ulaw',
                  output_audio_format: 'g711_ulaw',
                  voice: VOICE, // Use alloy voice
                  instructions: instruction, // Use the message defined above
                  modalities: ["text", "audio"], // Use text and audio modalities
                  temperature: 0.8, // Controls randomness of AI responses
                  input_audio_transcription: {'model': 'whisper-1'},
              }
          };
          openAiWs.send(JSON.stringify(sessionUpdate));
      };
      const sendInitialConversationItem = () => {
        const initialConversationItem = {
            type: "conversation.item.create",
            item: {
                type: "message",

                role: "user",
                content: [
                {
                    type: "input_text",
                    text: "Greet the user."
                }
            ]
          }
        };
        openAiWs.send(JSON.stringify(initialConversationItem));
        openAiWs.send(JSON.stringify({ type: "response.create" }));
      };

      // Open event for OpenAI WebSocket
      openAiWs.on('open', async () => {
          console.log('Connected to the OpenAI Realtime API');
          setTimeout(sendSessionUpdate, 250); // Ensure connection stability, send after .25 seconds
          setTimeout(sendInitialConversationItem, 250); // Send the initial conversation item
      });
      
      // Listen for messages from the OpenAI WebSocket (and send to Twilio if necessary)
      openAiWs.on('message', (data) => {
          try {
              const response = JSON.parse(data);
              // Get transcript of user's input
              if (response.type === 'conversation.item.input_audio_transcription.completed') {
                let before = callInformation.get(callDT).transcription;
                const cleanedTranscript = response.transcript.replace(/\n/g, ''); // Remove all newline characters
                const newTranscription = before + 'User: ' + cleanedTranscript + ' | ';
                callInformation.get(callDT).transcription = newTranscription;

              }
              // Get transcript of AI's response
              if (response.type === 'response.audio_transcript.done') {
                let before = callInformation.get(callDT).transcription;
                const cleanedTranscript = response.transcript.replace(/\n/g, ''); // Remove all newline characters
                const newTranscription = before + 'AI: ' + cleanedTranscript + ' | ';
                callInformation.get(callDT).transcription = newTranscription;
              }


              if (response.type === 'response.audio.delta' && response.delta) { // Handles AI-generated audio data from OpenAI, re-encodes it, and sends it to Twilio
                  const audioDelta = {
                      event: 'media',
                      streamSid: streamSid,
                      media: { payload: Buffer.from(response.delta, 'base64').toString('base64') }
                  };
                  connection.send(JSON.stringify(audioDelta));
              }
          } catch (error) {
              console.error('Error processing OpenAI message:', error, 'Raw message:', data);
          }
      });
      // Handle incoming messages from Twilio
      connection.on('message', (message) => {
          try {
              const data = JSON.parse(message);
              switch (data.event) {
                  case 'media': // processes and forwards audio data payloads from ongoing call to OpenAI
                      if (openAiWs.readyState === WebSocket.OPEN) {
                          const audioAppend = {
                              type: 'input_audio_buffer.append',
                              audio: data.media.payload
                          };
                          openAiWs.send(JSON.stringify(audioAppend));
                      }
                      break;
                  case 'start': // catches the stream's unique ID (streamSid)
                      streamSid = data.start.streamSid;
                      console.log('Incoming stream has started', streamSid);
                      break;
                  default:
                      console.log('Received non-media event:', data.event);
                      break;
              }
          } catch (error) {
              console.error('Error parsing message:', error, 'Message:', message);
          }
      });
      // Handle connection close
      connection.on('close', () => {
          if (openAiWs.readyState === WebSocket.OPEN) openAiWs.close();
          console.log('Client disconnected.');
      });
      // Handle WebSocket close and errors
      openAiWs.on('close', () => {
          console.log('Disconnected from the OpenAI Realtime API');
      });
      openAiWs.on('error', (error) => {
          console.error('Error in the OpenAI WebSocket:', error);
      });
  });
});

// Handle Twilio's POST status callbacks
fastify.post('/api/status-callback', async (request, reply) => {
  try {
    const { CallSid, CallStatus } = request.body;
    
    if (!CallSid || !CallStatus) {
      console.error('Missing required fields in Twilio callback:', request.body);
      return reply.code(400).send({ error: 'CallSid and CallStatus are required' });
    }
    
    // Store the status
    callStatuses.set(CallSid, CallStatus);
    console.log(`Call ${CallSid} status updated to: ${CallStatus}`);
    
    return reply.code(200).send();
  } catch (error) {
    console.error('Error in status-callback POST:', error);
    return reply.code(500).send({ error: 'Internal server error' });
  }
});

// Handle frontend GET status requests
fastify.get('/api/status-callback', async (request, reply) => {
  try {
    const { callSid } = request.query;
    
    if (!callSid) {
      console.error('Missing callSid in GET request');
      return reply.code(400).send({ error: 'CallSid is required' });
    }

    const status = callStatuses.get(callSid);
    
    if (!status) {
      console.log(`No status found for callSid: ${callSid}`);
      return reply.code(404).send({ error: 'Call status not found' });
    }

    return reply.send({ status });
  } catch (error) {
    console.error('Error in status-callback GET:', error);
    return reply.code(500).send({ error: 'Internal server error' });
  }
});

fastify.listen({ port: PORT }, (err) => {
  if (err) {
      console.error(err);
      process.exit(1);
  }
});

// Dev call for testing
(async () => {
  const result = await createCall('+17473347145', "You are a helpful and bubbly AI assistant who loves to chat about anything the user is interested about and is prepared to offer them facts. You have a penchant for dad jokes, owl jokes, and rickrolling – subtly. Always stay positive, but work in a joke when appropriate.");
  console.log(result.transcription); // This will now log the resolved value of callInfo
})();

export { createCall };