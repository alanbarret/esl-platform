import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Loader2, Play, RotateCcw, Download, Languages, Zap } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { AvatarViewer } from '../components/AvatarViewer';

export default function Home() {
  const {
    inputText, setInputText,
    language, setLanguage,
    outputFormat, setOutputFormat,
    isTranslating, glossTokens,
    videoUrl, gltfAnimation, error,
    translate, reset,
  } = useAppStore();

  const [activeTab, setActiveTab] = useState<'avatar' | 'video'>('avatar');

  return (
    <div className="min-h-screen bg-[#09090B] text-white font-sans">
      {/* Header */}
      <header className="border-b border-white/5 bg-black/40 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-violet-600 to-purple-500
                            flex items-center justify-center font-black text-sm">AI</div>
            <span className="font-black text-lg tracking-tight">
              ESL <span className="text-[#A8FF4B]">Platform</span>
            </span>
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <Zap size={12} className="text-[#A8FF4B]" />
            Emirati Sign Language AI
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-10 grid grid-cols-1 lg:grid-cols-2 gap-8">

        {/* Left: Input Panel */}
        <div className="space-y-5">
          <div>
            <h1 className="text-3xl font-black tracking-tight mb-1">
              Text to <span className="text-[#A8FF4B]">Sign Language</span>
            </h1>
            <p className="text-gray-400 text-sm">
              Convert Arabic or English text into Emirati Sign Language avatar video.
            </p>
          </div>

          {/* Language selector */}
          <div className="flex gap-2">
            {(['auto', 'ar', 'en'] as const).map((lang) => (
              <button
                key={lang}
                onClick={() => setLanguage(lang)}
                className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all
                  ${language === lang
                    ? 'bg-violet-600 text-white'
                    : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}
              >
                {lang === 'auto' ? 'Auto' : lang === 'ar' ? 'العربية' : 'English'}
              </button>
            ))}
          </div>

          {/* Text input */}
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            placeholder="اكتب نصًا بالعربية أو الإنجليزية..."
            dir="auto"
            rows={5}
            style={{ backgroundColor: '#1a1a2e', color: '#f1f1f1', caretColor: '#A8FF4B' }}
            className="w-full border border-white/10 rounded-xl px-4 py-3
                       text-sm placeholder-gray-500 resize-none outline-none
                       focus:border-violet-500 focus:ring-1 focus:ring-violet-500/40
                       transition-all font-arabic"
          />

          {/* Output format */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500 font-semibold uppercase tracking-wider">Output</span>
            {(['mp4', 'gltf'] as const).map((fmt) => (
              <button
                key={fmt}
                onClick={() => setOutputFormat(fmt)}
                className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all
                  ${outputFormat === fmt
                    ? 'bg-[#A8FF4B] text-black'
                    : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}
              >
                {fmt.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Generate button */}
          <div className="flex gap-3">
            <button
              onClick={translate}
              disabled={isTranslating || !inputText.trim()}
              className="flex-1 flex items-center justify-center gap-2 bg-[#A8FF4B] text-black
                         font-bold py-3 rounded-xl disabled:opacity-40 hover:bg-[#BFFF6E]
                         transition-all active:scale-[.98]"
            >
              {isTranslating ? (
                <><Loader2 size={16} className="animate-spin" /> Generating...</>
              ) : (
                <><Play size={16} /> Generate Sign Video</>
              )}
            </button>
            {(videoUrl || gltfAnimation || error) && (
              <button onClick={reset}
                className="px-4 py-3 bg-white/5 hover:bg-white/10 rounded-xl transition-all">
                <RotateCcw size={16} />
              </button>
            )}
          </div>

          {/* Error */}
          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 text-red-400 text-sm">
              {error}
            </div>
          )}

          {/* Gloss tokens */}
          {glossTokens.length > 0 && (
            <motion.div
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              className="bg-violet-500/10 border border-violet-500/20 rounded-xl p-4"
            >
              <div className="text-xs text-violet-400 font-semibold mb-2 uppercase tracking-wider">
                ESL Gloss Sequence
              </div>
              <div className="flex flex-wrap gap-2">
                {glossTokens.map((g, i) => (
                  <span key={i}
                    className="bg-violet-600/30 border border-violet-500/30 px-3 py-1
                               rounded-lg text-sm font-mono font-bold text-violet-200">
                    {g}
                  </span>
                ))}
              </div>
            </motion.div>
          )}
        </div>

        {/* Right: Output Panel */}
        <div className="space-y-4">
          {/* Tab switcher */}
          {(gltfAnimation || videoUrl) && (
            <div className="flex gap-2 border-b border-white/5 pb-4">
              <button onClick={() => setActiveTab('avatar')}
                className={`text-sm font-semibold pb-1 transition-all border-b-2
                  ${activeTab === 'avatar' ? 'border-[#A8FF4B] text-white' : 'border-transparent text-gray-500'}`}>
                3D Avatar
              </button>
              {videoUrl && (
                <button onClick={() => setActiveTab('video')}
                  className={`text-sm font-semibold pb-1 ml-4 transition-all border-b-2
                    ${activeTab === 'video' ? 'border-[#A8FF4B] text-white' : 'border-transparent text-gray-500'}`}>
                  Video
                </button>
              )}
            </div>
          )}

          {/* Avatar viewer */}
          {(gltfAnimation || isTranslating) && activeTab === 'avatar' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
              <AvatarViewer animation={gltfAnimation} className="aspect-[4/3]" />
            </motion.div>
          )}

          {/* Video player */}
          {videoUrl && activeTab === 'video' && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="relative bg-black rounded-2xl overflow-hidden aspect-video">
              <video src={videoUrl} controls autoPlay loop className="w-full h-full" />
              <a href={videoUrl} download
                className="absolute top-3 right-3 bg-black/60 backdrop-blur text-white
                           p-2 rounded-lg hover:bg-black/80 transition-all">
                <Download size={16} />
              </a>
            </motion.div>
          )}

          {/* Empty state */}
          {!gltfAnimation && !videoUrl && !isTranslating && (
            <div className="aspect-[4/3] bg-white/2 border border-white/6 rounded-2xl
                            flex flex-col items-center justify-center gap-3 text-gray-600">
              <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center
                              justify-center text-3xl">🤟</div>
              <p className="text-sm">Enter text and generate to see the avatar signing</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
