import { create } from 'zustand';
import { AppState, Language, OutputFormat } from '../types';
import type { MocapData } from '../components/AvatarViewer';

interface ESLState extends AppState {
  mocapData: MocapData | null;
  setMocapData: (d: MocapData | null) => void;
}

export const useAppStore = create<ESLState>((set, get) => ({
  // Input
  inputText: '',
  language: 'auto',
  outputFormat: 'gltf',

  // Pipeline
  isTranslating: false,
  glossTokens: [],
  currentGloss: '',

  // Output
  videoUrl: null,
  gltfAnimation: null,
  mocapData: null,
  error: null,

  // Actions
  setInputText: (text) => set({ inputText: text }),
  setLanguage: (lang: Language) => set({ language: lang }),
  setOutputFormat: (fmt: OutputFormat) => set({ outputFormat: fmt }),
  setMocapData: (d) => set({ mocapData: d }),

  translate: async () => {
    const { inputText } = get();
    if (!inputText.trim()) return;

    set({ isTranslating: true, error: null, videoUrl: null, gltfAnimation: null, mocapData: null, glossTokens: [] });

    try {
      // Step 1: Get gloss tokens from backend
      const glossRes = await fetch('/api/v1/gloss', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });
      const glossData = await glossRes.json();
      const tokens: string[] = glossData.gloss_tokens || [inputText.toUpperCase()];
      set({ glossTokens: tokens });

      // Step 2: For each gloss token, try to fetch real mocap data
      // Chain them: play the first one that has mocap, or fallback to keyframes
      const firstToken = tokens[0];
      const mocapRes = await fetch(`/api/v1/mocap/${firstToken}`);
      if (mocapRes.ok) {
        const mocapData: MocapData = await mocapRes.json();
        set({ mocapData, gltfAnimation: null, isTranslating: false });
        return;
      }

      // Step 3: Fallback — request keyframe animation (old system)
      const animRes = await fetch('/api/v1/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText, output_format: 'gltf' }),
      });
      const animData = await animRes.json();
      set({
        gltfAnimation: animData.gltf_animation ?? null,
        mocapData: null,
        isTranslating: false,
      });
    } catch (err: any) {
      set({ error: err.message || 'Translation failed', isTranslating: false });
    }
  },

  reset: () => set({
    inputText: '',
    glossTokens: [],
    videoUrl: null,
    gltfAnimation: null,
    mocapData: null,
    error: null,
    isTranslating: false,
  }),
}));
