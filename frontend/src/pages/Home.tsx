import React, { useState, useRef } from 'react';
import { motion } from 'framer-motion';
import { Loader2, Play, RotateCcw, ChevronLeft, ChevronRight, Zap } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import type { MocapData } from '../components/AvatarViewer';

export default function Home() {
  const {
    inputText, setInputText,
    language, setLanguage,
    isTranslating, glossTokens,
    skeletonVideos, videoUrl, avatarVideoUrl, error,
    translate, reset,
  } = useAppStore();

  const [currentVideoIdx, setCurrentVideoIdx] = useState(0);
  const [videoMode, setVideoMode] = useState<'skeleton'|'avatar'>('skeleton');
  const videoRef = useRef<HTMLVideoElement>(null);

  const allVideos = skeletonVideos.length > 0 ? skeletonVideos : (videoUrl ? [videoUrl] : []);
  const rawVideo = allVideos[currentVideoIdx] ?? null;
  // Avatar mode: use avatarVideoUrl directly (works for both typed text and dropdown)
  const currentVideo = videoMode === 'avatar'
    ? (avatarVideoUrl ?? rawVideo?.replace('/api/v1/skeleton-video/', '/api/v1/avatar-video/') ?? null)
    : rawVideo;
  // Reset index when video list changes
  React.useEffect(() => { setCurrentVideoIdx(0); }, [allVideos.length, allVideos[0]]);

  const loadMocap = (sign: string) => {
    const vidUrl = `/api/v1/skeleton-video/${sign}`;
    const avUrl = `/api/v1/avatar-video/${sign}`;
    setCurrentVideoIdx(0);
    useAppStore.setState({
      glossTokens: [sign],
      skeletonVideos: [vidUrl],
      videoUrl: vidUrl,
      avatarVideoUrl: avUrl,
      gltfAnimation: null,
      error: null,
      isTranslating: false,
    } as any);
  };

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

          {/* Test Pose Dropdown */}
          <div className="flex items-center gap-3">
            <span className="text-xs text-gray-500 font-semibold uppercase tracking-wider">Test Poses</span>
            <select
              defaultValue=""
              onChange={async (e) => {
                const pose = e.target.value;
                if (!pose) return;
                setInputText(pose);
                useAppStore.setState({ mocapData: null } as any);
                await loadMocap(pose);
              }}
              className="bg-[#1a1a2e] border border-white/10 rounded-lg px-3 py-2 text-sm text-white outline-none cursor-pointer hover:border-violet-500 transition-all flex-1"
            >
              <option value="">-- Select a sign --</option>
              <optgroup label="Common Signs">
                <option value="HOW_ARE_YOU">👋 How Are You</option>
                <option value="DOCTOR">🏥 Doctor</option>
                <option value="FAMILY">👨‍👩‍👧 Family</option>
                <option value="SCHOOL">🏫 School</option>
                <option value="WORK">💼 Work</option>
                <option value="MORNING">🌅 Morning</option>
                <option value="SLEEP">😴 Sleep</option>
                <option value="OPEN">📂 Open</option>
                <option value="PUSH">🤜 Push</option>
                <option value="HELP">🆘 Helps</option>
                <option value="RELAX">😌 Relax</option>
                <option value="RECOMMENDED">⭐ Recommended</option>
              </optgroup>
              <optgroup label="Actions">
                <option value="PLAYS">🎮 Plays</option>
                <option value="PLAYS_GUITAR">🎸 Plays Guitar</option>
                <option value="WATERING">💧 Watering</option>
                <option value="SEW">🧵 Sew</option>
                <option value="SENDS">📤 Sends</option>
                <option value="SELLS">🛒 Sells</option>
                <option value="RUSH">⚡ Rush</option>
                <option value="REMOVE">❌ Remove</option>
                <option value="PULLS">↙️ Pulls</option>
                <option value="PLOW">🌾 Plow</option>
                <option value="SHOUTS">📢 Shouts</option>
                <option value="RUBBING">🤲 Rubbing</option>
              </optgroup>
              <optgroup label="Arabic Alphabet">
                <option value="ALIF">ا Alif</option>
                <option value="BAA">ب Baa</option>
                <option value="TAA">ت Taa</option>
                <option value="JEEM">ج Jeem</option>
                <option value="HAA">ح Haa</option>
                <option value="DAAL">د Daal</option>
                <option value="RAA">ر Raa</option>
                <option value="SEEN">س Seen</option>
                <option value="SHEEN">ش Sheen</option>
                <option value="SAAD">ص Saad</option>
                <option value="DAAD">ض Daad</option>
                <option value="FAA">ف Faa</option>
                <option value="QAAF">ق Qaaf</option>
                <option value="KAAF">ك Kaaf</option>
                <option value="LAA">ل Laa</option>
                <option value="LAAM">لا Laam</option>
                <option value="MEEM">م Meem</option>
                <option value="NOON">ن Noon</option>
                <option value="YAA">ي Yaa</option>
              </optgroup>
            </select>
          </div>

          {/* Language selector */}
          <div className="flex gap-2">
            {(['auto', 'ar', 'en'] as const).map((lang) => (
              <button key={lang} onClick={() => setLanguage(lang)}
                className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-all
                  ${language === lang ? 'bg-violet-600 text-white' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}>
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
            {(videoUrl || error) && (
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
              <div className="text-xs text-violet-400 font-semibold mb-2 uppercase tracking-wider flex justify-between">
                <span>ESL Gloss Sequence</span>
                {avatarVideoUrl && <span className="text-[#A8FF4B]">✓ Avatar ready</span>}
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

        {/* Right: Video Player */}
        <div className="space-y-4">

          {/* Loading */}
          {isTranslating && (
            <div className="aspect-video bg-white/2 border border-white/6 rounded-2xl
                            flex items-center justify-center gap-3 text-gray-500">
              <Loader2 size={22} className="animate-spin text-violet-400" />
              <span className="text-sm">Generating sign video...</span>
            </div>
          )}

          {/* Video + mode toggle */}
          {currentVideo && !isTranslating && (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
              {/* Mode toggle */}
              <div className="flex gap-2">
                <button onClick={() => setVideoMode('skeleton')}
                  className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all
                    ${videoMode==='skeleton' ? 'bg-[#A8FF4B] text-black' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}>
                  🦴 Skeleton
                </button>
                <button onClick={() => setVideoMode('avatar')}
                  disabled={!avatarVideoUrl}
                  title={!avatarVideoUrl ? 'Avatar not available for this sign' : ''}
                  className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all
                    ${videoMode==='avatar' ? 'bg-[#A8FF4B] text-black' : 'bg-white/5 text-gray-400 hover:bg-white/10'}
                    disabled:opacity-40 disabled:cursor-not-allowed`}>
                  👳 Arab Avatar
                </button>
              </div>
              <div className="relative bg-black rounded-2xl overflow-hidden aspect-video border border-white/5">
                <video
                  ref={videoRef}
                  key={currentVideo}
                  src={currentVideo}
                  autoPlay loop muted
                  className="w-full h-full object-contain"
                />
                {/* Sign label badge */}
                <div className="absolute top-3 left-3 bg-black/70 backdrop-blur px-3 py-1 rounded-full
                                text-[#A8FF4B] font-bold text-xs border border-[#A8FF4B]/30">
                  {glossTokens[currentVideoIdx] || 'Sign'}
                </div>
              </div>

              {/* Multi-sign navigation */}
              {allVideos.length > 1 && (
                <div className="flex items-center justify-between bg-white/3 rounded-xl px-4 py-2">
                  <button
                    onClick={() => setCurrentVideoIdx(i => Math.max(0, i-1))}
                    disabled={currentVideoIdx === 0}
                    className="p-1 rounded-lg hover:bg-white/10 disabled:opacity-30 transition-all">
                    <ChevronLeft size={18} />
                  </button>
                  <div className="flex gap-2">
                    {allVideos.map((_, i) => (
                      <button key={i} onClick={() => setCurrentVideoIdx(i)}
                        className={`px-3 py-1 rounded-lg text-xs font-bold transition-all
                          ${i === currentVideoIdx ? 'bg-[#A8FF4B] text-black' : 'bg-white/5 text-gray-400 hover:bg-white/10'}`}>
                        {glossTokens[i] || i+1}
                      </button>
                    ))}
                  </div>
                  <button
                    onClick={() => setCurrentVideoIdx(i => Math.min(allVideos.length-1, i+1))}
                    disabled={currentVideoIdx === allVideos.length-1}
                    className="p-1 rounded-lg hover:bg-white/10 disabled:opacity-30 transition-all">
                    <ChevronRight size={18} />
                  </button>
                </div>
              )}
            </motion.div>
          )}

          {/* Empty state */}
          {!currentVideo && !isTranslating && (
            <div className="aspect-video bg-white/2 border border-white/6 rounded-2xl
                            flex flex-col items-center justify-center gap-3 text-gray-600">
              <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center
                              justify-center text-3xl">🤟</div>
              <p className="text-sm">Enter text or pick a sign to see the skeleton animation</p>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
