
import React, { useState } from 'react';
import { X, Zap, RotateCcw } from 'lucide-react';
import { Button } from '../ui/button';
import { motion } from 'motion/react';

interface CameraScreenProps {
  side: 'front' | 'back';
  onCapture: (image: string) => void;
  onBack: () => void;
}

const SAMPLE_FRONT = "https://images.unsplash.com/photo-1727157540259-51c72b9ac315?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=1080";
const SAMPLE_BACK = "https://images.unsplash.com/photo-1600196024905-e0cd65ddc6f1?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&q=80&w=1080";

export function CameraScreen({ side, onCapture, onBack }: CameraScreenProps) {
  const [isFlashing, setIsFlashing] = useState(false);

  const handleCapture = () => {
    setIsFlashing(true);
    setTimeout(() => {
      setIsFlashing(false);
      onCapture(side === 'front' ? SAMPLE_FRONT : SAMPLE_BACK);
    }, 300);
  };

  return (
    <div className="flex-1 bg-black relative flex flex-col">
      {/* Top Bar */}
      <div className="absolute top-0 left-0 right-0 p-6 flex justify-between items-center z-20 text-white">
        <button onClick={onBack} className="p-3 bg-black/40 rounded-full backdrop-blur-xl border border-white/10 hover:bg-white/10 transition-colors">
          <X className="w-5 h-5" />
        </button>
        <div className="bg-black/60 px-4 py-1.5 rounded-full backdrop-blur-xl border border-white/10">
          <span className="font-medium text-xs tracking-widest uppercase text-white/90">{side} of Card</span>
        </div>
        <button className="p-3 bg-black/40 rounded-full backdrop-blur-xl border border-white/10 hover:bg-white/10 transition-colors">
          <Zap className="w-5 h-5 text-yellow-400" />
        </button>
      </div>

      {/* Viewfinder Area */}
      <div className="flex-1 relative overflow-hidden flex items-center justify-center bg-slate-950">
        {/* Simulated Camera Feed Background */}
        <div className="absolute inset-0 bg-neutral-900 flex items-center justify-center">
            {/* Just a placeholder for the real camera feed */}
            <div className="text-center opacity-30">
                <p className="text-white text-xs mb-2 tracking-widest uppercase">Live Feed</p>
                <div className="w-1.5 h-1.5 bg-red-500 rounded-full animate-pulse mx-auto shadow-[0_0_10px_rgba(239,68,68,0.8)]"></div>
            </div>
        </div>

        {/* Card Outline Overlay - 2.5" x 3.5" Ratio (5:7) */}
        <div className="relative w-[75%] aspect-[5/7] border border-white/30 rounded-xl shadow-[0_0_0_9999px_rgba(0,0,0,0.85)] z-10">
              {/* Corner Guides */}
              <div className="absolute top-0 left-0 w-8 h-8 border-t-2 border-l-2 border-cyan-400 -mt-px -ml-px rounded-tl-lg shadow-[0_0_10px_rgba(34,211,238,0.5)]"></div>
              <div className="absolute top-0 right-0 w-8 h-8 border-t-2 border-r-2 border-cyan-400 -mt-px -mr-px rounded-tr-lg shadow-[0_0_10px_rgba(34,211,238,0.5)]"></div>
              <div className="absolute bottom-0 left-0 w-8 h-8 border-b-2 border-l-2 border-cyan-400 -mb-px -ml-px rounded-bl-lg shadow-[0_0_10px_rgba(34,211,238,0.5)]"></div>
              <div className="absolute bottom-0 right-0 w-8 h-8 border-b-2 border-r-2 border-cyan-400 -mb-px -mr-px rounded-br-lg shadow-[0_0_10px_rgba(34,211,238,0.5)]"></div>
              
              {/* Center Message */}
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="text-white/90 text-[10px] font-semibold bg-black/60 px-4 py-2 rounded-full backdrop-blur-md border border-white/10 uppercase tracking-wide">
                  Align {side}
                </p>
              </div>

              {/* Grid Lines (Subtle) */}
              <div className="absolute inset-0 grid grid-cols-3 grid-rows-3 pointer-events-none opacity-20">
                  <div className="border-r border-white/50"></div>
                  <div className="border-r border-white/50"></div>
                  <div></div>
                  <div className="border-t border-white/50 col-span-3"></div>
                  <div className="border-t border-white/50 col-span-3"></div>
              </div>

              {/* Dimension indicators */}
              <div className="absolute -left-8 top-1/2 -translate-y-1/2 -rotate-90 text-[9px] text-cyan-400/70 font-mono tracking-widest">3.5"</div>
              <div className="absolute bottom-[-28px] left-1/2 -translate-x-1/2 text-[9px] text-cyan-400/70 font-mono tracking-widest">2.5"</div>
        </div>

        {/* Flash Effect */}
        {isFlashing && (
          <div className="absolute inset-0 bg-white z-50 animate-out fade-out duration-300"></div>
        )}
      </div>

      {/* Bottom Controls */}
      <div className="h-36 bg-black flex items-center justify-center gap-10 pb-8 relative z-20 shrink-0">
        <button className="text-white/30 hover:text-white transition-colors p-4">
          <div className="w-12 h-12 rounded-xl bg-white/5 border border-white/10 backdrop-blur-md"></div>
        </button>
        
        <button 
          onClick={handleCapture}
          className="w-20 h-20 rounded-full border border-white/20 flex items-center justify-center group active:scale-95 transition-transform relative"
        >
          <div className="absolute inset-0 rounded-full bg-white/5 blur-md"></div>
          <div className="w-16 h-16 bg-white rounded-full group-hover:bg-cyan-400 transition-colors shadow-[0_0_20px_rgba(255,255,255,0.3)] z-10 border-4 border-black"></div>
        </button>

        <button className="text-white/30 hover:text-white transition-colors p-4">
          <RotateCcw className="w-6 h-6" />
        </button>
      </div>
    </div>
  );
}
