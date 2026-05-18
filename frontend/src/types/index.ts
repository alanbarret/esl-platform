// ESL Platform — TypeScript types

export type Language = 'auto' | 'ar' | 'en';
export type OutputFormat = 'mp4' | 'gltf' | 'json';

export interface TranslateRequest {
  text: string;
  language: Language;
  output_format: OutputFormat;
  fps: number;
  width: number;
  height: number;
  transparent_bg: boolean;
}

export interface TranslateResponse {
  request_id: string;
  input_text: string;
  detected_language: Language;
  gloss_tokens: string[];
  total_duration: number;
  status: 'completed' | 'failed' | 'processing';
  video_url?: string;
  gltf_animation?: GLTFAnimation;
  error?: string;
}

export interface GLTFAnimation {
  name: string;
  channels: AnimationChannel[];
  samplers: AnimationSampler[];
  duration: number;
  fps: number;
}

export interface AnimationChannel {
  sampler: number;
  target: { node: string; path: 'rotation' | 'translation' | 'scale' };
}

export interface AnimationSampler {
  input: number[];
  interpolation: 'LINEAR' | 'STEP' | 'CUBICSPLINE';
  output: number[];
}

export interface ModelStatus {
  gloss_model: {
    loaded: boolean;
    device: string;
  };
}

export interface AppState {
  // Input
  inputText: string;
  language: Language;
  outputFormat: OutputFormat;

  // Pipeline state
  isTranslating: boolean;
  glossTokens: string[];
  currentGloss: string;

  // Output
  videoUrl: string | null;
  gltfAnimation: GLTFAnimation | null;
  error: string | null;

  // Actions
  setInputText: (text: string) => void;
  setLanguage: (lang: Language) => void;
  setOutputFormat: (fmt: OutputFormat) => void;
  translate: () => Promise<void>;
  reset: () => void;
}
