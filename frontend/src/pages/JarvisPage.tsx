/**
 * JarvisPage.tsx — Tony Stark's J.A.R.V.I.S. Interface
 *
 * A dedicated voice-first interface inspired by Iron Man's AI assistant.
 * Features:
 * - Wake word detection ("Hey Jarvis")
 * - Voice responses ("Yes sir")
 * - HUD-style animated visualization
 * - Full voice conversation loop
 * - Voice Activity Detection (VAD) for natural turn-taking
 * - Barge-in support (interrupt Jarvis while speaking)
 */

import { useState, useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router';
import { ArrowLeft, Mic, MicOff, Volume2, VolumeX } from 'lucide-react';
import { toast } from 'sonner';
import { useAppStore, generateId } from '../lib/store';
import { streamChat } from '../lib/sse';
import { getBase, transcribeAudio, fetchSpeechHealth } from '../lib/api';
import type { ChatMessage, TokenUsage, ToolCallInfo } from '../types';

type JarvisState = 'idle' | 'listening' | 'processing' | 'speaking' | 'error';

const WAKE_PHRASES = ['hey jarvis', 'jarvis', 'hey j.a.r.v.i.s', 'hello jarvis', 'ok jarvis'];
const ACKNOWLEDGMENTS = [
  'Yes sir.',
  'At your service, sir.',
  'How may I assist you?',
  'I\'m here, sir.',
  'Ready when you are, sir.',
];

// VAD Configuration
const VAD_CONFIG = {
  silenceThreshold: 0.02,      // RMS threshold below which is considered silence
  silenceDuration: 800,        // ms of silence before stopping (natural pause detection)
  minRecordingTime: 500,       // Minimum recording time in ms to avoid false triggers
  speechStartThreshold: 0.03,  // RMS threshold to detect speech has started
};

export function JarvisPage() {
  const navigate = useNavigate();
  const [state, setState] = useState<JarvisState>('idle');
  const [transcript, setTranscript] = useState('');
  const [response, setResponse] = useState('');
  const [wakeWordEnabled, setWakeWordEnabled] = useState(false); // Disabled by default - use manual mic
  const [ttsEnabled, setTtsEnabled] = useState(true);
  const [visualizerBars, setVisualizerBars] = useState<number[]>(Array(32).fill(0.1));
  const [speechAvailable, setSpeechAvailable] = useState(false);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationRef = useRef<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const recognitionRef = useRef<any>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const stateRef = useRef<JarvisState>('idle');
  const isRecognitionActiveRef = useRef(false);
  const restartTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Streaming TTS refs
  const ttsQueueRef = useRef<string[]>([]);
  const isSpeakingRef = useRef(false);
  const pendingTextRef = useRef('');
  const streamingCompleteRef = useRef(false); // Track when LLM streaming is done
  const shouldAutoListenRef = useRef(false); // Flag to trigger auto-listen after speaking

  // VAD (Voice Activity Detection) refs
  const vadIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const recordingStartRef = useRef<number | null>(null);
  const speechDetectedRef = useRef(false);

  // Barge-in refs (interrupt while speaking)
  const bargeInStreamRef = useRef<MediaStream | null>(null);
  const bargeInAnalyserRef = useRef<AnalyserNode | null>(null);
  const bargeInContextRef = useRef<AudioContext | null>(null);
  const bargeInCheckRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Conversation context for more natural dialogue
  const conversationHistoryRef = useRef<Array<{ role: 'user' | 'assistant'; content: string }>>([]);
  const lastInteractionTimeRef = useRef<number>(Date.now());

  // Barge-in handler ref (set later to avoid circular dependency)
  const bargeInHandlerRef = useRef<(() => Promise<void>) | null>(null);

  const selectedModel = useAppStore((s) => s.selectedModel);
  const activeId = useAppStore((s) => s.activeId);
  const createConversation = useAppStore((s) => s.createConversation);
  const addMessage = useAppStore((s) => s.addMessage);
  const updateLastAssistant = useAppStore((s) => s.updateLastAssistant);

  // Keep stateRef in sync
  useEffect(() => {
    stateRef.current = state;
  }, [state]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Clean up VAD
      if (vadIntervalRef.current) {
        clearInterval(vadIntervalRef.current);
      }
      // Clean up barge-in detection
      if (bargeInCheckRef.current) {
        clearInterval(bargeInCheckRef.current);
      }
      if (bargeInStreamRef.current) {
        bargeInStreamRef.current.getTracks().forEach(track => track.stop());
      }
      if (bargeInContextRef.current) {
        bargeInContextRef.current.close().catch(() => {});
      }
      // Clean up audio resources
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (audioContextRef.current) {
        audioContextRef.current.close().catch(() => {});
      }
    };
  }, []);

  // Auto-listen after speaking completes (continuous conversation)
  useEffect(() => {
    if (state === 'idle' && shouldAutoListenRef.current) {
      shouldAutoListenRef.current = false;
      // Small delay before starting to listen again for natural conversation flow
      const timer = setTimeout(() => {
        if (stateRef.current === 'idle') {
          // Re-use the startListening logic - we'll trigger it via a custom event
          window.dispatchEvent(new CustomEvent('jarvis-auto-listen'));
        }
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [state]);

  // Check speech backend availability
  useEffect(() => {
    fetchSpeechHealth()
      .then((health) => setSpeechAvailable(health.available))
      .catch(() => setSpeechAvailable(false));
  }, []);

  // Animate visualizer bars
  useEffect(() => {
    const animate = () => {
      if (analyserRef.current && (state === 'listening' || state === 'speaking')) {
        const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
        analyserRef.current.getByteFrequencyData(dataArray);
        const bars = Array.from({ length: 32 }, (_, i) => {
          const index = Math.floor((i / 32) * dataArray.length);
          return Math.max(0.1, dataArray[index] / 255);
        });
        setVisualizerBars(bars);
      } else if (state === 'processing') {
        // Pulsing animation during processing
        setVisualizerBars(prev => prev.map((_, i) =>
          0.3 + 0.7 * Math.abs(Math.sin(Date.now() / 200 + i * 0.2))
        ));
      } else {
        // Idle breathing animation
        setVisualizerBars(prev => prev.map((_, i) =>
          0.1 + 0.15 * Math.abs(Math.sin(Date.now() / 1000 + i * 0.15))
        ));
      }
      animationRef.current = requestAnimationFrame(animate);
    };
    animationRef.current = requestAnimationFrame(animate);
    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
    };
  }, [state]);

  // Synthesize and play a single sentence
  const speakSentence = useCallback(async (text: string): Promise<void> => {
    if (!text.trim()) return;

    try {
      const res = await fetch(`${getBase()}/v1/speech/synthesize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice: 'jarvis' }),
      });

      if (!res.ok) {
        // Fallback to browser TTS
        return new Promise((resolve) => {
          const utterance = new SpeechSynthesisUtterance(text);
          utterance.rate = 0.95;
          utterance.pitch = 0.9;
          utterance.onend = () => resolve();
          utterance.onerror = () => resolve();
          window.speechSynthesis.speak(utterance);
        });
      }

      const audioBlob = await res.blob();
      const audioUrl = URL.createObjectURL(audioBlob);

      return new Promise((resolve) => {
        if (audioRef.current) {
          audioRef.current.src = audioUrl;
          audioRef.current.onended = () => {
            URL.revokeObjectURL(audioUrl);
            resolve();
          };
          audioRef.current.onerror = () => {
            URL.revokeObjectURL(audioUrl);
            resolve();
          };
          audioRef.current.play().catch(() => resolve());
        } else {
          resolve();
        }
      });
    } catch {
      // Fallback to browser TTS
      return new Promise((resolve) => {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 0.95;
        utterance.pitch = 0.9;
        utterance.onend = () => resolve();
        utterance.onerror = () => resolve();
        window.speechSynthesis.speak(utterance);
      });
    }
  }, []);

  // Stop barge-in detection
  const stopBargeInDetection = useCallback(() => {
    if (bargeInCheckRef.current) {
      clearInterval(bargeInCheckRef.current);
      bargeInCheckRef.current = null;
    }
    if (bargeInStreamRef.current) {
      bargeInStreamRef.current.getTracks().forEach(track => track.stop());
      bargeInStreamRef.current = null;
    }
    if (bargeInContextRef.current) {
      bargeInContextRef.current.close().catch(() => {});
      bargeInContextRef.current = null;
    }
    bargeInAnalyserRef.current = null;
  }, []);

  // Start barge-in detection (listens for user speech while Jarvis is speaking)
  const startBargeInDetection = useCallback(async (onBargeIn: () => void) => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      bargeInStreamRef.current = stream;

      const audioContext = new AudioContext();
      bargeInContextRef.current = audioContext;

      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      bargeInAnalyserRef.current = analyser;

      const source = audioContext.createMediaStreamSource(stream);
      source.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      let consecutiveSpeechFrames = 0;
      const BARGE_IN_THRESHOLD = 0.08; // Higher than silence to avoid false triggers
      const FRAMES_REQUIRED = 3; // Require sustained speech

      bargeInCheckRef.current = setInterval(() => {
        if (!bargeInAnalyserRef.current) return;

        analyser.getByteFrequencyData(dataArray);
        // Calculate RMS
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          const normalized = dataArray[i] / 255;
          sum += normalized * normalized;
        }
        const rms = Math.sqrt(sum / dataArray.length);

        if (rms > BARGE_IN_THRESHOLD) {
          consecutiveSpeechFrames++;
          if (consecutiveSpeechFrames >= FRAMES_REQUIRED) {
            // User is speaking - trigger barge-in
            stopBargeInDetection();
            onBargeIn();
          }
        } else {
          consecutiveSpeechFrames = 0;
        }
      }, 50); // Check every 50ms
    } catch {
      // Microphone access denied or error - continue without barge-in
    }
  }, [stopBargeInDetection]);

  // Calculate RMS from analyser
  const calculateRMS = useCallback((): number => {
    if (!analyserRef.current) return 0;
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount);
    analyserRef.current.getByteFrequencyData(dataArray);
    let sum = 0;
    for (let i = 0; i < dataArray.length; i++) {
      const normalized = dataArray[i] / 255;
      sum += normalized * normalized;
    }
    return Math.sqrt(sum / dataArray.length);
  }, []);

  // Stop VAD monitoring
  const stopVAD = useCallback(() => {
    if (vadIntervalRef.current) {
      clearInterval(vadIntervalRef.current);
      vadIntervalRef.current = null;
    }
    silenceStartRef.current = null;
    recordingStartRef.current = null;
    speechDetectedRef.current = false;
  }, []);

  // Start VAD monitoring for natural turn-taking
  const startVAD = useCallback((onSilenceDetected: () => void) => {
    recordingStartRef.current = Date.now();
    silenceStartRef.current = null;
    speechDetectedRef.current = false;

    vadIntervalRef.current = setInterval(() => {
      const rms = calculateRMS();
      const now = Date.now();
      const recordingDuration = now - (recordingStartRef.current || now);

      // Detect if speech has started
      if (!speechDetectedRef.current && rms > VAD_CONFIG.speechStartThreshold) {
        speechDetectedRef.current = true;
        silenceStartRef.current = null;
      }

      // Only check for silence after speech has been detected and minimum time passed
      if (speechDetectedRef.current && recordingDuration >= VAD_CONFIG.minRecordingTime) {
        if (rms < VAD_CONFIG.silenceThreshold) {
          // Below silence threshold
          if (silenceStartRef.current === null) {
            silenceStartRef.current = now;
          } else if (now - silenceStartRef.current >= VAD_CONFIG.silenceDuration) {
            // Silence duration exceeded - user stopped speaking
            stopVAD();
            onSilenceDetected();
          }
        } else {
          // Above threshold - reset silence timer
          silenceStartRef.current = null;
        }
      }
    }, 50); // Check every 50ms
  }, [calculateRMS, stopVAD]);

  // Process TTS queue - speaks sentences one by one
  const processQueue = useCallback(async () => {
    if (isSpeakingRef.current) return;
    if (ttsQueueRef.current.length === 0) {
      // Check if we're done (no pending text and streaming finished)
      if (pendingTextRef.current === '' && streamingCompleteRef.current && stateRef.current === 'speaking') {
        // Stop barge-in detection when done speaking
        stopBargeInDetection();
        // Set flag to trigger auto-listen, then go idle (effect will pick this up)
        shouldAutoListenRef.current = true;
        setState('idle');
      }
      return;
    }

    isSpeakingRef.current = true;
    setState('speaking');

    // Start barge-in detection while speaking (allows user to interrupt)
    startBargeInDetection(() => {
      if (bargeInHandlerRef.current) {
        bargeInHandlerRef.current();
      }
    });

    while (ttsQueueRef.current.length > 0) {
      const sentence = ttsQueueRef.current.shift()!;
      await speakSentence(sentence);
    }

    isSpeakingRef.current = false;

    // Check if more sentences arrived while we were speaking
    if (ttsQueueRef.current.length > 0) {
      processQueue();
    } else if (pendingTextRef.current === '' && streamingCompleteRef.current && stateRef.current === 'speaking') {
      // Stop barge-in detection when done speaking
      stopBargeInDetection();
      // Set flag to trigger auto-listen, then go idle
      shouldAutoListenRef.current = true;
      setState('idle');
    }
  }, [speakSentence, startBargeInDetection, stopBargeInDetection]);

  // Queue text for streaming TTS - extracts complete sentences
  const queueTextForTTS = useCallback((newText: string, isComplete: boolean = false) => {
    if (!ttsEnabled) {
      if (isComplete) {
        streamingCompleteRef.current = true;
        // If TTS is disabled, just go idle after streaming completes
        shouldAutoListenRef.current = true;
        setState('idle');
      }
      return;
    }

    pendingTextRef.current += newText;

    // Extract complete sentences (ending with . ! ?)
    const sentenceRegex = /[^.!?]*[.!?]+/g;
    let match;
    let lastIndex = 0;

    while ((match = sentenceRegex.exec(pendingTextRef.current)) !== null) {
      const sentence = match[0].trim();
      if (sentence) {
        ttsQueueRef.current.push(sentence);
      }
      lastIndex = sentenceRegex.lastIndex;
    }

    // Keep remaining incomplete text
    pendingTextRef.current = pendingTextRef.current.slice(lastIndex);

    // If streaming is complete and there's remaining text, queue it
    if (isComplete) {
      streamingCompleteRef.current = true;
      if (pendingTextRef.current.trim()) {
        ttsQueueRef.current.push(pendingTextRef.current.trim());
        pendingTextRef.current = '';
      }
    }

    // Start processing queue if not already speaking
    processQueue();
  }, [ttsEnabled, processQueue]);

  // Reset TTS state
  const resetTTS = useCallback(() => {
    ttsQueueRef.current = [];
    pendingTextRef.current = '';
    isSpeakingRef.current = false;
    streamingCompleteRef.current = false;
    shouldAutoListenRef.current = false;
  }, []);

  // Legacy speak function for single utterances (acknowledgments)
  const speak = useCallback(async (text: string) => {
    if (!ttsEnabled) {
      setState('idle');
      return;
    }
    setState('speaking');
    await speakSentence(text);
    setState('idle');
  }, [ttsEnabled, speakSentence]);

  // Handle barge-in (user interrupts while Jarvis is speaking)
  // Note: This will be called via ref to avoid circular dependencies
  const handleBargeIn = useCallback(async () => {
    // Stop current speech
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    window.speechSynthesis?.cancel();

    // Clear TTS queue
    resetTTS();
    stopBargeInDetection();

    // Abort any ongoing LLM request
    if (abortRef.current) {
      abortRef.current.abort();
      abortRef.current = null;
    }

    // Set state to trigger listening
    setState('idle');
    // Dispatch event to trigger auto-listen with small delay
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('jarvis-auto-listen'));
    }, 100);
  }, [resetTTS, stopBargeInDetection]);

  // Set the barge-in handler ref (avoids circular dependency)
  useEffect(() => {
    bargeInHandlerRef.current = handleBargeIn;
  }, [handleBargeIn]);

  // Process voice command through LLM
  const processCommand = useCallback(async (command: string) => {
    if (!command.trim() || !selectedModel) {
      setState('idle');
      return;
    }

    setState('processing');
    setResponse('');
    resetTTS(); // Clear any previous TTS state

    // Update conversation history for context
    conversationHistoryRef.current.push({ role: 'user', content: command });
    lastInteractionTimeRef.current = Date.now();

    let convId = activeId;
    if (!convId) {
      convId = createConversation(selectedModel);
    }

    const userMsg: ChatMessage = {
      id: generateId(),
      role: 'user',
      content: command,
      timestamp: Date.now(),
    };
    addMessage(convId, userMsg);

    const assistantMsg: ChatMessage = {
      id: generateId(),
      role: 'assistant',
      content: '',
      timestamp: Date.now(),
    };
    addMessage(convId, assistantMsg);

    // Enhanced system prompt for more natural conversation
    const systemPrompt = `You are J.A.R.V.I.S., Tony Stark's AI assistant. You have a warm, British butler-like demeanor.

Key traits:
- Be concise and conversational (this is voice output, keep responses 1-3 sentences when possible)
- Address the user as "sir" naturally, but not in every sentence
- Show genuine interest and personality
- Remember context from the conversation
- If the user seems frustrated or repeats themselves, acknowledge it naturally
- Vary your acknowledgments (don't always say the same phrases)

Previous exchanges for context:
${conversationHistoryRef.current.slice(-6).map(m => `${m.role === 'user' ? 'User' : 'Jarvis'}: ${m.content}`).join('\n')}`;

    const messages = [
      { role: 'system' as const, content: systemPrompt },
      { role: 'user' as const, content: command }
    ];
    const controller = new AbortController();
    abortRef.current = controller;

    let accumulatedContent = '';
    let usage: TokenUsage | undefined;
    const toolCalls: ToolCallInfo[] = [];

    try {
      for await (const sseEvent of streamChat(
        {
          model: selectedModel,
          messages,
          stream: true,
        },
        controller.signal
      )) {
        try {
          const data = JSON.parse(sseEvent.data);
          if (data.usage) usage = data.usage;
          const delta = data.choices?.[0]?.delta?.content || '';
          if (delta) {
            accumulatedContent += delta;
            setResponse(accumulatedContent);
            // Stream text to TTS as it arrives
            queueTextForTTS(delta, false);
          }
          if (data.choices?.[0]?.finish_reason === 'stop') break;
        } catch {}
      }
    } catch (err: any) {
      if (err.name !== 'AbortError') {
        accumulatedContent = 'I apologize sir, but I encountered an error processing your request.';
        setResponse(accumulatedContent);
        queueTextForTTS(accumulatedContent, true);
      }
    } finally {
      updateLastAssistant(convId, accumulatedContent, toolCalls.length > 0 ? toolCalls : undefined, usage);
      abortRef.current = null;

      // Update conversation history
      if (accumulatedContent) {
        conversationHistoryRef.current.push({ role: 'assistant', content: accumulatedContent });
        // Keep only last 10 exchanges to avoid token bloat
        if (conversationHistoryRef.current.length > 20) {
          conversationHistoryRef.current = conversationHistoryRef.current.slice(-20);
        }
      }

      // Signal TTS that streaming is complete (flush any remaining text)
      if (accumulatedContent) {
        queueTextForTTS('', true);
      } else {
        setState('idle');
      }
    }
  }, [selectedModel, activeId, createConversation, addMessage, updateLastAssistant, queueTextForTTS, resetTTS]);

  // Start listening for voice input
  const startListening = useCallback(async () => {
    try {
      setState('listening');
      setTranscript('');

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;

      // Set up audio analysis for visualizer
      audioContextRef.current = new AudioContext();
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);

      const recorder = new MediaRecorder(stream);
      audioChunksRef.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) audioChunksRef.current.push(e.data);
      };

      recorder.start();
      mediaRecorderRef.current = recorder;
    } catch (err) {
      toast.error('Microphone access denied');
      setState('error');
      setTimeout(() => setState('idle'), 2000);
    }
  }, []);

  // Stop listening and process
  const stopListening = useCallback(async () => {
    return new Promise<string>((resolve) => {
      const recorder = mediaRecorderRef.current;
      if (!recorder || recorder.state !== 'recording') {
        setState('idle');
        resolve('');
        return;
      }

      recorder.onstop = async () => {
        setState('processing');

        streamRef.current?.getTracks().forEach(track => track.stop());
        streamRef.current = null;

        if (audioContextRef.current) {
          try {
            await audioContextRef.current.close();
          } catch {}
          audioContextRef.current = null;
        }

        const blob = new Blob(audioChunksRef.current, { type: recorder.mimeType || 'audio/webm' });
        audioChunksRef.current = [];

        try {
          const result = await transcribeAudio(blob);
          setTranscript(result.text);
          resolve(result.text);
        } catch (err) {
          toast.error('Transcription failed');
          setState('error');
          setTimeout(() => setState('idle'), 2000);
          resolve('');
        }
      };

      recorder.stop();
    });
  }, []);

  // Listen for auto-listen events (triggered after speaking completes)
  useEffect(() => {
    const handleAutoListen = async () => {
      if (stateRef.current === 'idle') {
        await startListening();

        // Use VAD for natural turn-taking instead of fixed timeout
        startVAD(async () => {
          if (mediaRecorderRef.current?.state === 'recording') {
            const command = await stopListening();
            if (command) {
              await processCommand(command);
            }
          }
        });

        // Fallback: max 10 seconds of listening to prevent hanging
        setTimeout(async () => {
          stopVAD();
          if (mediaRecorderRef.current?.state === 'recording') {
            const command = await stopListening();
            if (command) {
              await processCommand(command);
            }
          }
        }, 10000);
      }
    };

    window.addEventListener('jarvis-auto-listen', handleAutoListen);
    return () => window.removeEventListener('jarvis-auto-listen', handleAutoListen);
  }, [startListening, stopListening, processCommand, startVAD, stopVAD]);

  // Initialize wake word detection using Web Speech API
  useEffect(() => {
    if (!wakeWordEnabled) {
      // Clean up if disabled
      if (recognitionRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {}
        recognitionRef.current = null;
      }
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
        restartTimeoutRef.current = null;
      }
      isRecognitionActiveRef.current = false;
      return;
    }

    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      console.warn('Speech recognition not supported in this browser');
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.continuous = false; // Use non-continuous mode to avoid restart issues
    recognition.interimResults = false;
    recognition.lang = 'en-US';
    recognition.maxAlternatives = 1;

    recognition.onresult = async (event: any) => {
      if (!event.results || event.results.length === 0) return;

      const text = event.results[0][0].transcript.toLowerCase().trim();
      console.log('Heard:', text);

      // Check for wake phrase
      if (stateRef.current === 'idle' && WAKE_PHRASES.some(phrase => text.includes(phrase))) {
        isRecognitionActiveRef.current = false;

        // Acknowledge wake word
        const ack = ACKNOWLEDGMENTS[Math.floor(Math.random() * ACKNOWLEDGMENTS.length)];
        setResponse(ack);
        await speak(ack);

        // Start listening for command
        await startListening();

        // Use VAD for natural turn-taking instead of fixed timeout
        startVAD(async () => {
          if (mediaRecorderRef.current?.state === 'recording') {
            const command = await stopListening();
            if (command) {
              await processCommand(command);
            }
          }
        });

        // Fallback: max 10 seconds to prevent hanging
        setTimeout(async () => {
          stopVAD();
          if (mediaRecorderRef.current?.state === 'recording') {
            const command = await stopListening();
            if (command) {
              await processCommand(command);
            }
          }
        }, 10000);
      }
    };

    recognition.onerror = (event: any) => {
      // Ignore common non-fatal errors
      if (event.error === 'no-speech' || event.error === 'aborted') {
        return;
      }
      console.warn('Speech recognition error:', event.error);
    };

    recognition.onend = () => {
      isRecognitionActiveRef.current = false;

      // Only restart if wake word is still enabled and we're idle
      if (wakeWordEnabled && stateRef.current === 'idle') {
        // Debounce restart to prevent rapid cycling
        if (restartTimeoutRef.current) {
          clearTimeout(restartTimeoutRef.current);
        }
        restartTimeoutRef.current = setTimeout(() => {
          if (wakeWordEnabled && stateRef.current === 'idle' && !isRecognitionActiveRef.current) {
            try {
              recognition.start();
              isRecognitionActiveRef.current = true;
            } catch (e) {
              // Already running or other error
            }
          }
        }, 500);
      }
    };

    // Initial start
    try {
      recognition.start();
      isRecognitionActiveRef.current = true;
    } catch (e) {
      console.warn('Failed to start speech recognition:', e);
    }

    recognitionRef.current = recognition;

    return () => {
      if (restartTimeoutRef.current) {
        clearTimeout(restartTimeoutRef.current);
        restartTimeoutRef.current = null;
      }
      try {
        recognition.stop();
      } catch {}
      isRecognitionActiveRef.current = false;
    };
  }, [wakeWordEnabled, startListening, stopListening, processCommand, speak, startVAD, stopVAD]);

  // Handle manual mic button click
  const handleMicClick = useCallback(async () => {
    if (state === 'listening') {
      // Stop VAD and process
      stopVAD();
      const command = await stopListening();
      if (command) {
        await processCommand(command);
      }
    } else if (state === 'idle') {
      // Stop wake word detection while manually recording
      if (recognitionRef.current && isRecognitionActiveRef.current) {
        try {
          recognitionRef.current.stop();
        } catch {}
        isRecognitionActiveRef.current = false;
      }
      await startListening();

      // Start VAD for automatic stop on silence
      startVAD(async () => {
        if (mediaRecorderRef.current?.state === 'recording') {
          const command = await stopListening();
          if (command) {
            await processCommand(command);
          }
        }
      });

      // Fallback: max 10 seconds
      setTimeout(async () => {
        stopVAD();
        if (mediaRecorderRef.current?.state === 'recording') {
          const command = await stopListening();
          if (command) {
            await processCommand(command);
          }
        }
      }, 10000);
    } else if (state === 'speaking') {
      // Allow interrupting Jarvis while speaking (manual barge-in)
      handleBargeIn();
    }
  }, [state, startListening, stopListening, processCommand, startVAD, stopVAD, handleBargeIn]);

  // Get state-based colors
  const getStateColor = () => {
    switch (state) {
      case 'listening': return { primary: '#22d3ee', rgb: '34, 211, 238' };
      case 'processing': return { primary: '#f59e0b', rgb: '245, 158, 11' };
      case 'speaking': return { primary: '#22c55e', rgb: '34, 197, 94' };
      case 'error': return { primary: '#ef4444', rgb: '239, 68, 68' };
      default: return { primary: '#22d3ee', rgb: '34, 211, 238' };
    }
  };
  const stateColor = getStateColor();

  return (
    <div className="fixed inset-0 flex flex-col overflow-hidden" style={{ background: 'linear-gradient(135deg, #0a0a0f 0%, #0f1419 50%, #0a0a0f 100%)' }}>
      {/* Hidden audio element for TTS playback */}
      <audio ref={audioRef} />

      {/* Animated Background */}
      <div className="absolute inset-0 pointer-events-none overflow-hidden">
        {/* Animated grid */}
        <div
          className="absolute inset-0"
          style={{
            backgroundImage: `
              linear-gradient(rgba(${stateColor.rgb}, 0.03) 1px, transparent 1px),
              linear-gradient(90deg, rgba(${stateColor.rgb}, 0.03) 1px, transparent 1px)
            `,
            backgroundSize: '60px 60px',
            animation: 'gridMove 20s linear infinite',
          }}
        />

        {/* Floating particles */}
        {Array.from({ length: 20 }).map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full"
            style={{
              width: `${2 + Math.random() * 4}px`,
              height: `${2 + Math.random() * 4}px`,
              left: `${Math.random() * 100}%`,
              top: `${Math.random() * 100}%`,
              background: `rgba(${stateColor.rgb}, ${0.1 + Math.random() * 0.3})`,
              animation: `float ${10 + Math.random() * 20}s ease-in-out infinite`,
              animationDelay: `${Math.random() * 10}s`,
            }}
          />
        ))}

        {/* Central glow that responds to state */}
        <div
          className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full"
          style={{
            background: `radial-gradient(circle, rgba(${stateColor.rgb}, ${state !== 'idle' ? 0.15 : 0.05}) 0%, transparent 70%)`,
            transition: 'all 0.5s ease-out',
            animation: state !== 'idle' ? 'breathe 2s ease-in-out infinite' : 'none',
          }}
        />

        {/* Scan lines effect */}
        <div
          className="absolute inset-0 opacity-[0.02]"
          style={{
            backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.03) 2px, rgba(255,255,255,0.03) 4px)',
          }}
        />

        {/* HUD corner brackets with glow */}
        <svg className="absolute top-6 left-6 w-24 h-24" style={{ filter: `drop-shadow(0 0 8px rgba(${stateColor.rgb}, 0.5))` }}>
          <path d="M0 30 L0 0 L30 0" fill="none" stroke={stateColor.primary} strokeWidth="2" opacity="0.6" />
          <circle cx="0" cy="0" r="3" fill={stateColor.primary} opacity="0.8" />
        </svg>
        <svg className="absolute top-6 right-6 w-24 h-24" style={{ filter: `drop-shadow(0 0 8px rgba(${stateColor.rgb}, 0.5))` }}>
          <path d="M66 0 L96 0 L96 30" fill="none" stroke={stateColor.primary} strokeWidth="2" opacity="0.6" />
          <circle cx="96" cy="0" r="3" fill={stateColor.primary} opacity="0.8" />
        </svg>
        <svg className="absolute bottom-6 left-6 w-24 h-24" style={{ filter: `drop-shadow(0 0 8px rgba(${stateColor.rgb}, 0.5))` }}>
          <path d="M0 66 L0 96 L30 96" fill="none" stroke={stateColor.primary} strokeWidth="2" opacity="0.6" />
          <circle cx="0" cy="96" r="3" fill={stateColor.primary} opacity="0.8" />
        </svg>
        <svg className="absolute bottom-6 right-6 w-24 h-24" style={{ filter: `drop-shadow(0 0 8px rgba(${stateColor.rgb}, 0.5))` }}>
          <path d="M66 96 L96 96 L96 66" fill="none" stroke={stateColor.primary} strokeWidth="2" opacity="0.6" />
          <circle cx="96" cy="96" r="3" fill={stateColor.primary} opacity="0.8" />
        </svg>
      </div>

      {/* Header */}
      <header className="relative z-10 flex items-center justify-between px-6 py-4">
        <button
          onClick={() => navigate('/')}
          className="group flex items-center gap-2 px-4 py-2 rounded-xl transition-all duration-300 text-cyan-400 hover:text-cyan-300 hover:bg-cyan-400/10 hover:scale-105"
        >
          <ArrowLeft size={18} className="transition-transform group-hover:-translate-x-1" />
          <span className="text-sm font-medium">Back</span>
        </button>

        <div className="relative">
          <h1
            className="text-2xl tracking-[0.4em] font-light"
            style={{
              color: stateColor.primary,
              fontFamily: 'var(--font-display)',
              textShadow: `0 0 30px rgba(${stateColor.rgb}, 0.6), 0 0 60px rgba(${stateColor.rgb}, 0.3)`,
              transition: 'all 0.3s ease',
            }}
          >
            J.A.R.V.I.S.
          </h1>
          <div
            className="absolute -bottom-1 left-0 right-0 h-px"
            style={{
              background: `linear-gradient(90deg, transparent, rgba(${stateColor.rgb}, 0.5), transparent)`,
            }}
          />
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setTtsEnabled(!ttsEnabled)}
            className="relative p-3 rounded-xl transition-all duration-300 hover:scale-110"
            style={{
              color: ttsEnabled ? stateColor.primary : '#5c5c63',
              background: ttsEnabled ? `rgba(${stateColor.rgb}, 0.1)` : 'rgba(255,255,255,0.02)',
              boxShadow: ttsEnabled ? `0 0 20px rgba(${stateColor.rgb}, 0.2)` : 'none',
            }}
            title={ttsEnabled ? 'Voice enabled' : 'Voice disabled'}
          >
            {ttsEnabled ? <Volume2 size={20} /> : <VolumeX size={20} />}
          </button>
          <button
            onClick={() => setWakeWordEnabled(!wakeWordEnabled)}
            className="relative p-3 rounded-xl transition-all duration-300 hover:scale-110"
            style={{
              color: wakeWordEnabled ? stateColor.primary : '#5c5c63',
              background: wakeWordEnabled ? `rgba(${stateColor.rgb}, 0.1)` : 'rgba(255,255,255,0.02)',
              boxShadow: wakeWordEnabled ? `0 0 20px rgba(${stateColor.rgb}, 0.2)` : 'none',
            }}
            title={wakeWordEnabled ? 'Wake word enabled' : 'Wake word disabled'}
          >
            {wakeWordEnabled ? <Mic size={20} /> : <MicOff size={20} />}
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="relative z-10 flex-1 flex flex-col items-center justify-center px-8">
        {/* Enhanced Arc Reactor Visualization */}
        <div className="relative w-80 h-80 mb-10">
          {/* Outer pulsing rings */}
          <div
            className="absolute inset-0 rounded-full"
            style={{
              border: `2px solid rgba(${stateColor.rgb}, 0.2)`,
              animation: 'pulse 2s ease-in-out infinite',
              boxShadow: `inset 0 0 30px rgba(${stateColor.rgb}, 0.1), 0 0 30px rgba(${stateColor.rgb}, 0.1)`,
            }}
          />
          <div
            className="absolute inset-4 rounded-full"
            style={{
              border: `1px solid rgba(${stateColor.rgb}, 0.15)`,
              animation: state !== 'idle' ? 'spin 8s linear infinite' : 'none',
            }}
          />
          <div
            className="absolute inset-8 rounded-full"
            style={{
              border: `1px dashed rgba(${stateColor.rgb}, 0.1)`,
              animation: state !== 'idle' ? 'spin 12s linear infinite reverse' : 'none',
            }}
          />

          {/* Hexagonal tech pattern overlay */}
          <div
            className="absolute inset-6 rounded-full opacity-20"
            style={{
              backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='52' viewBox='0 0 60 52' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 0l25.98 15v30L30 60 4.02 45V15z' fill='none' stroke='%2322d3ee' stroke-width='0.5'/%3E%3C/svg%3E")`,
              backgroundSize: '20px 20px',
              animation: state !== 'idle' ? 'spin 30s linear infinite' : 'none',
            }}
          />

          {/* Audio visualizer - more dynamic waveform */}
          <div className="absolute inset-10 rounded-full">
            <svg viewBox="0 0 200 200" className="w-full h-full">
              {visualizerBars.map((height, i) => {
                const angle = (i / visualizerBars.length) * 360 - 90;
                const innerRadius = 50;
                const outerRadius = 50 + height * 40;
                const x1 = 100 + innerRadius * Math.cos(angle * Math.PI / 180);
                const y1 = 100 + innerRadius * Math.sin(angle * Math.PI / 180);
                const x2 = 100 + outerRadius * Math.cos(angle * Math.PI / 180);
                const y2 = 100 + outerRadius * Math.sin(angle * Math.PI / 180);
                return (
                  <line
                    key={i}
                    x1={x1}
                    y1={y1}
                    x2={x2}
                    y2={y2}
                    stroke={stateColor.primary}
                    strokeWidth="2"
                    strokeLinecap="round"
                    opacity={0.3 + height * 0.7}
                    style={{ transition: 'all 50ms ease-out' }}
                  />
                );
              })}
              {/* Inner glow circle */}
              <circle
                cx="100"
                cy="100"
                r="45"
                fill="none"
                stroke={stateColor.primary}
                strokeWidth="1"
                opacity="0.3"
              />
            </svg>
          </div>

          {/* Center core with enhanced glow */}
          <div
            className="absolute inset-[85px] rounded-full flex items-center justify-center"
            style={{
              background: `radial-gradient(circle, rgba(${stateColor.rgb}, ${state !== 'idle' ? 0.4 : 0.2}) 0%, rgba(${stateColor.rgb}, 0.1) 60%, transparent 100%)`,
              boxShadow: `0 0 ${state !== 'idle' ? 80 : 40}px rgba(${stateColor.rgb}, ${state !== 'idle' ? 0.5 : 0.2}), inset 0 0 30px rgba(${stateColor.rgb}, 0.2)`,
              transition: 'all 0.4s ease',
            }}
          >
            <div
              className="w-12 h-12 rounded-full flex items-center justify-center"
              style={{
                backgroundColor: `rgba(${stateColor.rgb}, ${state !== 'idle' ? 0.9 : 0.6})`,
                boxShadow: `0 0 40px ${stateColor.primary}, 0 0 80px rgba(${stateColor.rgb}, 0.5)`,
                transition: 'all 0.3s ease',
                animation: state !== 'idle' ? 'corePulse 1s ease-in-out infinite' : 'none',
              }}
            >
              {state === 'processing' && (
                <div className="w-6 h-6 border-2 border-white/30 border-t-white rounded-full animate-spin" />
              )}
            </div>
          </div>

          {/* Orbiting elements when active */}
          {state !== 'idle' && (
            <>
              <div
                className="absolute w-3 h-3 rounded-full"
                style={{
                  background: stateColor.primary,
                  boxShadow: `0 0 10px ${stateColor.primary}`,
                  top: '50%',
                  left: '50%',
                  animation: 'orbit 3s linear infinite',
                }}
              />
              <div
                className="absolute w-2 h-2 rounded-full"
                style={{
                  background: stateColor.primary,
                  boxShadow: `0 0 8px ${stateColor.primary}`,
                  top: '50%',
                  left: '50%',
                  animation: 'orbit 4s linear infinite reverse',
                  animationDelay: '-1s',
                }}
              />
            </>
          )}
        </div>

        {/* Status indicator with animation */}
        <div className="text-center mb-8">
          <div
            className="inline-flex items-center gap-3 px-6 py-3 rounded-full mb-4"
            style={{
              background: `rgba(${stateColor.rgb}, 0.1)`,
              border: `1px solid rgba(${stateColor.rgb}, 0.2)`,
              boxShadow: `0 0 20px rgba(${stateColor.rgb}, 0.1)`,
            }}
          >
            <div
              className="w-2 h-2 rounded-full"
              style={{
                backgroundColor: stateColor.primary,
                boxShadow: `0 0 10px ${stateColor.primary}`,
                animation: state !== 'idle' ? 'blink 1s ease-in-out infinite' : 'none',
              }}
            />
            <p className="text-sm uppercase tracking-[0.2em] font-medium" style={{ color: stateColor.primary }}>
              {state === 'idle' && wakeWordEnabled && 'Awaiting Command'}
              {state === 'idle' && !wakeWordEnabled && 'Ready'}
              {state === 'listening' && 'Listening'}
              {state === 'processing' && 'Processing'}
              {state === 'speaking' && 'Speaking'}
              {state === 'error' && 'Error'}
            </p>
          </div>

          {transcript && (
            <div
              className="mt-4 px-6 py-3 rounded-xl max-w-lg mx-auto"
              style={{
                background: 'rgba(34, 211, 238, 0.05)',
                border: '1px solid rgba(34, 211, 238, 0.1)',
              }}
            >
              <p className="text-cyan-400 text-lg italic">"{transcript}"</p>
            </div>
          )}
        </div>

        {/* Response area with typewriter effect */}
        {response && (
          <div
            className="max-w-2xl mx-auto mb-8 p-8 rounded-2xl relative overflow-hidden"
            style={{
              backgroundColor: 'rgba(0, 0, 0, 0.4)',
              border: `1px solid rgba(${stateColor.rgb}, 0.2)`,
              boxShadow: `0 0 40px rgba(${stateColor.rgb}, 0.1), inset 0 0 60px rgba(0,0,0,0.5)`,
            }}
          >
            {/* Decorative corner accents */}
            <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 rounded-tl-lg" style={{ borderColor: stateColor.primary, opacity: 0.5 }} />
            <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 rounded-tr-lg" style={{ borderColor: stateColor.primary, opacity: 0.5 }} />
            <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 rounded-bl-lg" style={{ borderColor: stateColor.primary, opacity: 0.5 }} />
            <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 rounded-br-lg" style={{ borderColor: stateColor.primary, opacity: 0.5 }} />

            <p
              className="text-lg leading-relaxed"
              style={{
                color: '#ededef',
                textShadow: '0 0 20px rgba(255,255,255,0.1)',
              }}
            >
              {response}
              {state === 'speaking' && <span className="inline-block w-2 h-5 bg-cyan-400 ml-1 animate-pulse" />}
            </p>
          </div>
        )}

        {/* Enhanced mic button */}
        <div className="relative">
          {/* Ripple effects when active */}
          {state === 'listening' && (
            <>
              <div className="absolute inset-0 rounded-full animate-ping" style={{ background: `rgba(${stateColor.rgb}, 0.3)` }} />
              <div className="absolute -inset-4 rounded-full animate-pulse" style={{ border: `2px solid rgba(${stateColor.rgb}, 0.2)` }} />
            </>
          )}

          <button
            onClick={handleMicClick}
            disabled={state === 'processing'}
            className="relative w-20 h-20 rounded-full flex items-center justify-center transition-all duration-300 disabled:opacity-50 cursor-pointer hover:scale-110 active:scale-95"
            style={{
              background: `linear-gradient(145deg, rgba(${stateColor.rgb}, 0.2), rgba(${stateColor.rgb}, 0.05))`,
              border: `2px solid ${stateColor.primary}`,
              boxShadow: `0 0 40px rgba(${stateColor.rgb}, 0.3), inset 0 0 20px rgba(${stateColor.rgb}, 0.1)`,
            }}
          >
            {state === 'listening' ? (
              <div
                className="w-7 h-7 rounded-md transition-all duration-300"
                style={{ backgroundColor: '#ef4444', boxShadow: '0 0 20px #ef4444' }}
              />
            ) : state === 'processing' ? (
              <div className="w-8 h-8 border-3 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <Mic size={28} style={{ color: stateColor.primary, filter: `drop-shadow(0 0 10px ${stateColor.primary})` }} />
            )}
          </button>
        </div>

        <p className="mt-4 text-sm font-medium tracking-wide" style={{ color: stateColor.primary, opacity: 0.7 }}>
          {state === 'listening' ? 'Tap to stop' : state === 'speaking' ? 'Tap to interrupt' : state === 'processing' ? 'Processing...' : 'Tap to speak'}
        </p>
      </main>

      {/* Footer status bar */}
      <footer className="relative z-10 px-6 py-4">
        <div
          className="flex items-center justify-center gap-8 px-6 py-3 rounded-full mx-auto max-w-fit"
          style={{
            background: 'rgba(0,0,0,0.4)',
            border: '1px solid rgba(34, 211, 238, 0.1)',
          }}
        >
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-xs font-medium tracking-wider" style={{ color: '#8d8d93' }}>
              {selectedModel ? selectedModel.split('/').pop()?.toUpperCase() : 'NO MODEL'}
            </span>
          </div>
          <div className="w-px h-4 bg-cyan-400/20" />
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${speechAvailable ? 'bg-green-400' : 'bg-red-400'}`} />
            <span className="text-xs font-medium tracking-wider" style={{ color: '#8d8d93' }}>
              SPEECH {speechAvailable ? 'ONLINE' : 'OFFLINE'}
            </span>
          </div>
          <div className="w-px h-4 bg-cyan-400/20" />
          <div className="flex items-center gap-2">
            <div className={`w-2 h-2 rounded-full ${wakeWordEnabled ? 'bg-cyan-400 animate-pulse' : 'bg-gray-500'}`} />
            <span className="text-xs font-medium tracking-wider" style={{ color: '#8d8d93' }}>
              WAKE {wakeWordEnabled ? 'ACTIVE' : 'OFF'}
            </span>
          </div>
        </div>
      </footer>

      {/* Enhanced CSS Animations */}
      <style>{`
        @keyframes spin {
          from { transform: rotate(0deg); }
          to { transform: rotate(360deg); }
        }
        @keyframes pulse {
          0%, 100% { opacity: 0.3; transform: scale(1); }
          50% { opacity: 0.6; transform: scale(1.03); }
        }
        @keyframes breathe {
          0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 0.8; }
          50% { transform: translate(-50%, -50%) scale(1.1); opacity: 1; }
        }
        @keyframes corePulse {
          0%, 100% { transform: scale(1); }
          50% { transform: scale(1.1); }
        }
        @keyframes blink {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.3; }
        }
        @keyframes float {
          0%, 100% { transform: translateY(0) translateX(0); opacity: 0.3; }
          25% { transform: translateY(-20px) translateX(10px); opacity: 0.6; }
          50% { transform: translateY(-10px) translateX(-5px); opacity: 0.4; }
          75% { transform: translateY(-30px) translateX(15px); opacity: 0.5; }
        }
        @keyframes gridMove {
          from { transform: translateY(0); }
          to { transform: translateY(60px); }
        }
        @keyframes orbit {
          from { transform: translate(-50%, -50%) rotate(0deg) translateX(120px) rotate(0deg); }
          to { transform: translate(-50%, -50%) rotate(360deg) translateX(120px) rotate(-360deg); }
        }
      `}</style>
    </div>
  );
}
