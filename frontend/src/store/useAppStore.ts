import { create } from 'zustand';
import { AppState, Language, OutputFormat } from '../types';
import type { MocapData } from '../components/AvatarViewer';

interface ESLState extends AppState {
  mocapData: MocapData | null;
  skeletonVideos: string[];
  avatarVideoUrl: string | null;
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
  skeletonVideos: [],
  avatarVideoUrl: null,
  error: null,

  // Actions
  setInputText: (text) => set({ inputText: text }),
  setLanguage: (lang: Language) => set({ language: lang }),
  setOutputFormat: (fmt: OutputFormat) => set({ outputFormat: fmt }),
  setMocapData: (d) => set({ mocapData: d }),

  translate: async () => {
    const { inputText } = get();
    if (!inputText.trim()) return;

    set({ isTranslating: true, error: null, videoUrl: null, avatarVideoUrl: null, gltfAnimation: null, mocapData: null, glossTokens: [], skeletonVideos: [] });

    try {
      // Call translate — backend returns skeleton_videos array
      const res = await fetch('/api/v1/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });
      const data = await res.json();
      const tokens: string[] = data.gloss_tokens || [inputText.toUpperCase()];
      set({
        glossTokens: tokens,
        skeletonVideos: data.skeleton_videos || [],
        videoUrl: data.skeleton_videos?.[0] || null,
        avatarVideoUrl: data.avatar_video_url || null,
        gltfAnimation: null,
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
    skeletonVideos: [],
    avatarVideoUrl: null,
    error: null,
    isTranslating: false,
  }),
}));
