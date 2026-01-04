
import React, { useEffect, useState } from 'react';
import { Button } from '../ui/button';
import { Card, CardContent } from '../ui/card';
import { ScanLine, ArrowRight, CheckCircle2, History, ChevronRight, User } from 'lucide-react';
import { motion } from 'motion/react';

// --- MOCK API SERVICE ---
const MockAPI = {
  fetchRecentScans: async () => {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 800));
    return [
      { id: 1, title: "Topps Chrome #23", date: "2h ago", grade: "9.5", img: "https://images.unsplash.com/photo-1534063228518-a6644df1714d?w=100&h=100&fit=crop" },
      { id: 2, title: "Panini Prizm #10", date: "5h ago", grade: "8.0", img: "https://images.unsplash.com/photo-1613771404784-3a5686aa2be3?w=100&h=100&fit=crop" },
      { id: 3, title: "Upper Deck #45", date: "1d ago", grade: "Pending", img: "https://images.unsplash.com/photo-1599508704512-2f19efd1e35f?w=100&h=100&fit=crop" },
    ];
  }
};

interface HomeScreenProps {
  frontImage: string | null;
  backImage: string | null;
  onScanFront: () => void;
  onScanBack: () => void;
  onAnalyze: () => void;
}

export function HomeScreen({ frontImage, backImage, onScanFront, onScanBack, onAnalyze }: HomeScreenProps) {
  const [recentScans, setRecentScans] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const readyToAnalyze = frontImage && backImage;

  useEffect(() => {
    MockAPI.fetchRecentScans().then(data => {
      setRecentScans(data);
      setLoading(false);
    });
  }, []);

  return (
    <div className="flex-1 flex flex-col text-white">
      {/* Header */}
      <header className="px-6 py-6 flex justify-between items-center sticky top-0 z-10 backdrop-blur-sm bg-slate-950/50 border-b border-white/5">
        <div>
          <p className="text-cyan-400 text-xs font-medium tracking-widest uppercase mb-1">Welcome Back</p>
          <h1 className="text-xl font-bold text-white tracking-tight">TruScore <span className="font-light text-white/50">System</span></h1>
        </div>
        <Button variant="ghost" size="icon" className="rounded-full bg-white/5 hover:bg-white/10 text-white border border-white/10">
          <User className="h-5 w-5" />
        </Button>
      </header>

      <div className="flex-1 p-6 space-y-8">
        
        {/* Main Scan Actions */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 mb-2">
             <div className="h-1.5 w-1.5 rounded-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.8)]"></div>
             <h2 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">New Submission</h2>
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            {/* Front Scan Button */}
            <motion.div whileTap={{ scale: 0.98 }}>
              <button 
                onClick={onScanFront}
                className={`w-full aspect-[4/5] rounded-2xl flex flex-col items-center justify-center p-4 transition-all relative overflow-hidden backdrop-blur-md ${
                  frontImage 
                    ? 'border-2 border-emerald-500/50 bg-emerald-900/20' 
                    : 'border border-white/10 bg-white/5 hover:bg-white/10 hover:border-cyan-500/30'
                }`}
              >
                {frontImage ? (
                  <>
                    <img src={frontImage} alt="Front" className="absolute inset-0 w-full h-full object-cover opacity-60" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent z-10" />
                    <div className="z-20 bg-emerald-500/20 border border-emerald-500/50 p-2 rounded-full shadow-[0_0_15px_rgba(16,185,129,0.3)] backdrop-blur-md">
                      <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                    </div>
                    <p className="z-20 mt-3 font-semibold text-white text-sm tracking-wide">Front Ready</p>
                  </>
                ) : (
                  <>
                    <div className="w-12 h-12 rounded-full bg-cyan-500/10 border border-cyan-500/20 flex items-center justify-center mb-4 shadow-[0_0_15px_rgba(6,182,212,0.1)]">
                      <ScanLine className="w-5 h-5 text-cyan-400" />
                    </div>
                    <span className="font-semibold text-white">Scan Front</span>
                    <span className="text-[10px] text-slate-400 mt-1 uppercase tracking-widest">Required</span>
                  </>
                )}
              </button>
            </motion.div>

            {/* Back Scan Button */}
            <motion.div whileTap={{ scale: 0.98 }}>
              <button 
                onClick={onScanBack}
                className={`w-full aspect-[4/5] rounded-2xl flex flex-col items-center justify-center p-4 transition-all relative overflow-hidden backdrop-blur-md ${
                  backImage 
                    ? 'border-2 border-emerald-500/50 bg-emerald-900/20' 
                    : 'border border-white/10 bg-white/5 hover:bg-white/10 hover:border-purple-500/30'
                }`}
              >
                {backImage ? (
                  <>
                    <img src={backImage} alt="Back" className="absolute inset-0 w-full h-full object-cover opacity-60" />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/80 to-transparent z-10" />
                    <div className="z-20 bg-emerald-500/20 border border-emerald-500/50 p-2 rounded-full shadow-[0_0_15px_rgba(16,185,129,0.3)] backdrop-blur-md">
                      <CheckCircle2 className="w-6 h-6 text-emerald-400" />
                    </div>
                    <p className="z-20 mt-3 font-semibold text-white text-sm tracking-wide">Back Ready</p>
                  </>
                ) : (
                  <>
                    <div className="w-12 h-12 rounded-full bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mb-4 shadow-[0_0_15px_rgba(168,85,247,0.1)]">
                      <ScanLine className="w-5 h-5 text-purple-400" />
                    </div>
                    <span className="font-semibold text-white">Scan Back</span>
                    <span className="text-[10px] text-slate-400 mt-1 uppercase tracking-widest">Required</span>
                  </>
                )}
              </button>
            </motion.div>
          </div>

          {/* Analyze Action */}
          {(frontImage || backImage) && (
            <motion.div 
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              <Button 
                onClick={onAnalyze}
                disabled={!readyToAnalyze}
                className={`w-full h-14 rounded-xl text-base font-bold shadow-lg transition-all border ${
                  readyToAnalyze 
                    ? 'bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 text-white border-cyan-400/30 shadow-[0_0_20px_rgba(6,182,212,0.3)]' 
                    : 'bg-white/5 text-slate-500 border-white/5 cursor-not-allowed'
                }`}
              >
                {readyToAnalyze ? (
                  <>
                    Analyze & Grade
                    <ArrowRight className="ml-2 w-5 h-5" />
                  </>
                ) : (
                  'Scan both sides to continue'
                )}
              </Button>
            </motion.div>
          )}
        </section>

        {/* Recent Activity */}
        <section>
          <div className="flex justify-between items-center mb-4">
             <div className="flex items-center gap-2">
                <History className="w-4 h-4 text-slate-400" />
                <h3 className="text-sm font-semibold text-slate-300 uppercase tracking-wider">Recent Scans</h3>
             </div>
             <Button variant="link" className="text-xs text-cyan-400 p-0 h-auto hover:text-cyan-300">
               View All
             </Button>
          </div>
          
          <div className="space-y-3">
             {loading ? (
                // Loading Skeletons
                [1, 2].map((i) => (
                  <div key={i} className="h-20 rounded-xl bg-white/5 animate-pulse border border-white/5"></div>
                ))
             ) : (
                recentScans.map((item) => (
                   <motion.div 
                     key={item.id}
                     whileHover={{ scale: 1.02 }}
                     className="flex items-center gap-4 p-3 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 hover:border-white/20 transition-all cursor-pointer group backdrop-blur-sm"
                   >
                     <div className="h-12 w-12 bg-slate-800 rounded-lg overflow-hidden flex-shrink-0 border border-white/10 relative">
                       <img src={item.img} alt={item.title} className="h-full w-full object-cover" />
                       <div className="absolute inset-0 bg-black/20 group-hover:bg-transparent transition-colors"></div>
                     </div>
                     <div className="flex-1">
                       <p className="text-sm font-semibold text-white group-hover:text-cyan-400 transition-colors">{item.title}</p>
                       <p className="text-xs text-slate-500">{item.date}</p>
                     </div>
                     <div className="flex flex-col items-end gap-1">
                        <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${
                            item.grade === 'Pending' 
                              ? 'bg-yellow-500/10 text-yellow-500 border-yellow-500/20' 
                              : 'bg-emerald-500/10 text-emerald-500 border-emerald-500/20'
                        }`}>
                            {item.grade}
                        </span>
                     </div>
                     <ChevronRight className="h-4 w-4 text-slate-600 group-hover:text-white transition-colors" />
                   </motion.div>
                ))
             )}
          </div>
        </section>
      </div>
    </div>
  );
}
