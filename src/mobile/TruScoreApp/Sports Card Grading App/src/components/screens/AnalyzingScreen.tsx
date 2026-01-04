
import React, { useEffect, useState } from 'react';
import { Progress } from '../ui/progress';
import { motion } from 'motion/react';
import { Search, Scan, Database, Cpu, Layers } from 'lucide-react';

interface AnalyzingScreenProps {
  onComplete: () => void;
}

const STEPS = [
  { text: "Scanning surface topography...", icon: Layers },
  { text: "Analyzing centering geometry...", icon: Scan },
  { text: "Detecting edge imperfections...", icon: Search },
  { text: "Querying global market database...", icon: Database },
  { text: "Calculating final TruScore...", icon: Cpu },
];

export function AnalyzingScreen({ onComplete }: AnalyzingScreenProps) {
  const [progress, setProgress] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(timer);
          return 100;
        }
        return prev + 1.5;
      });
    }, 50);

    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const newIndex = Math.min(Math.floor((progress / 100) * STEPS.length), STEPS.length - 1);
    setStepIndex(newIndex);

    if (progress >= 100) {
      setTimeout(onComplete, 800);
    }
  }, [progress, onComplete]);

  const CurrentIcon = STEPS[stepIndex].icon;

  return (
    <div className="flex-1 flex flex-col items-center justify-center bg-slate-950 p-8 relative overflow-hidden">
      {/* Background Effects */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[150%] h-[150%] bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-indigo-900/20 via-slate-950 to-slate-950 pointer-events-none"></div>

      <div className="z-10 w-full max-w-xs text-center space-y-10">
        <div className="relative mx-auto w-40 h-40 flex items-center justify-center">
             {/* Outer Ring */}
             <motion.div 
               animate={{ rotate: 360 }}
               transition={{ duration: 8, repeat: Infinity, ease: "linear" }}
               className="absolute inset-0 rounded-full border border-white/5 border-t-cyan-500/50 border-r-cyan-500/20"
             />
             {/* Inner Ring */}
             <motion.div 
               animate={{ rotate: -360 }}
               transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
               className="absolute inset-4 rounded-full border border-white/5 border-b-purple-500/50 border-l-purple-500/20"
             />
             
             {/* Glowing Core */}
             <div className="absolute inset-0 rounded-full bg-cyan-500/5 blur-3xl animate-pulse"></div>

             {/* Icon */}
             <motion.div
                key={stepIndex}
                initial={{ scale: 0.8, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                transition={{ type: "spring", stiffness: 200, damping: 20 }}
                className="relative z-10"
             >
                <CurrentIcon className="w-12 h-12 text-white drop-shadow-[0_0_15px_rgba(34,211,238,0.5)]" />
             </motion.div>
        </div>
        
        <div className="space-y-3">
            <h2 className="text-2xl font-bold text-white tracking-tight">System Processing</h2>
            <div className="h-6 overflow-hidden">
                <motion.p 
                    key={stepIndex}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="text-cyan-400 font-mono text-sm tracking-wide"
                >
                    {STEPS[stepIndex].text}
                </motion.p>
            </div>
        </div>

        <div className="space-y-3">
            <div className="h-1.5 w-full bg-white/5 rounded-full overflow-hidden backdrop-blur-sm">
                <motion.div 
                    className="h-full bg-gradient-to-r from-cyan-500 to-purple-600 shadow-[0_0_10px_rgba(34,211,238,0.5)]"
                    style={{ width: `${progress}%` }}
                />
            </div>
            <p className="text-[10px] text-slate-500 font-mono text-right w-full">{Math.round(progress)}% COMPLETE</p>
        </div>
      </div>
    </div>
  );
}
