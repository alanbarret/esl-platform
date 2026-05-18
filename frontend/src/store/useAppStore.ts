import { create } from 'zustand';
import { AppState, Language, OutputFormat } from '../types';
import { api } from '../utils/api';

export const useAppStore = create<AppState>((set, get) => ({
  // Input
  inputText: '',
  language: 'auto',
  outputFormat: 'mp4',

  // Pipeline
  isTranslating: false,
  glossTokens: [],
  currentGloss: '',

  // Output
  videoUrl: null,
  gltfAnimation: null,
  error: null,

  // Actions
  setInputText: (text) => set({ inputText: text }),
  setLanguage: (lang: Language) => set({ language: lang }),
  setOutputFormat: (fmt: OutputFormat) => set({ outputFormat: fmt }),

  translate: async () => {
    const { inputText, language, outputFormat } = get();
    if (!inputText.trim()) return;

    set({ isTranslating: true, error: null, videoUrl: null, gltfAnimation: null, glossTokens: [] });

    try {
      const result = await api.translate({
        text: inputText,
        language,
        output_format: outputFormat,
        fps: 30,
        width: 1920,
        height: 1080,
        transparent_bg: false,
      });

      set({
        glossTokens: result.gloss_tokens,
        videoUrl: result.video_url ? `http://localhost:8000${result.video_url}` : null,
        gltfAnimation: result.gltf_animation ?? null,
        isTranslating: false,
      });
    } catch (err: any) {
      set({
        error: err.message || 'Translation failed',
        isTranslating: false,
      });
    }
  },

  reset: () => set({
    inputText: '',
    glossTokens: [],
    videoUrl: null,
    gltfAnimation: null,
    error: null,
    isTranslating: false,
  }),
}));
