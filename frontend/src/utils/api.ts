import axios from 'axios';
import type { TranslateRequest, TranslateResponse, ModelStatus } from '../types';

const BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1';

const client = axios.create({ baseURL: BASE, timeout: 120_000 });

export const api = {
  translate: async (req: TranslateRequest): Promise<TranslateResponse> => {
    const { data } = await client.post<TranslateResponse>('/translate', req);
    return data;
  },

  glossOnly: async (text: string, language = 'auto') => {
    const { data } = await client.post('/gloss', { text, language });
    return data as { gloss_tokens: string[]; gloss_string: string };
  },

  modelsStatus: async (): Promise<ModelStatus> => {
    const { data } = await client.get<ModelStatus>('/models/status');
    return data;
  },

  health: async () => {
    const { data } = await client.get('/health');
    return data;
  },
};
