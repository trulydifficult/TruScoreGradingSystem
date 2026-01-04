
import React from 'react';
import { Button } from '../ui/button';
import { Card, CardContent } from '../ui/card';
import { Separator } from '../ui/separator';
import { Share2, Home, CheckCircle2, TrendingUp, Sparkles } from 'lucide-react';
import { motion } from 'motion/react';

interface ResultsScreenProps {
  frontImage: string | null;
  onHome: () => void;
}

export function ResultsScreen({ frontImage, onHome }: ResultsScreenProps) {
  return (
    <div className="flex-1 flex flex-col bg-slate-950 text-white">
      <div className="relative h-72 w-full flex items-center justify-center overflow-hidden shrink-0">
         {/* Background blurred image */}
         <div className="absolute inset-0">
            {frontImage && <img src={frontImage} className="w-full h-full object-cover opacity-20 blur-3xl scale-150" />}
            <div className="absolute inset-0 bg-gradient-to-b from-slate-950/0 via-slate-950/50 to-slate-950"></div>
         </div>
         
         {/* Main Card Display */}
         <motion.div 
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
            className="z-10 w-40 rounded-xl overflow-hidden shadow-2xl border border-white/20 relative group"
         >
            <div className="absolute inset-0 bg-gradient-to-tr from-white/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity"></div>
            {frontImage && <img src={frontImage} className="w-full h-auto" />}
            
            {/* Holographic Shine Effect */}
            <div className="absolute inset-0 bg-gradient-to-tr from-transparent via-white/10 to-transparent -translate-x-full animate-[shimmer_2s_infinite]"></div>
         </motion.div>

         <div className="absolute top-6 right-6 z-20">
             <Button variant="ghost" size="icon" className="text-white hover:bg-white/10 rounded-full border border-white/10 backdrop-blur-md">
                 <Share2 className="w-5 h-5" />
             </Button>
         </div>
      </div>

      <div className="flex-1 -mt-8 rounded-t-[2.5rem] bg-slate-950/80 backdrop-blur-xl border-t border-white/10 relative z-20 px-6 py-8 overflow-y-auto ring-1 ring-white/5">
         <div className="flex flex-col items-center mb-8">
             <motion.div 
               initial={{ scale: 0.8, opacity: 0 }}
               animate={{ scale: 1, opacity: 1 }}
               className="bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-4 py-1.5 rounded-full text-xs font-bold flex items-center gap-2 mb-4 uppercase tracking-wider shadow-[0_0_15px_rgba(16,185,129,0.2)]"
             >
                 <CheckCircle2 className="w-4 h-4" />
                 Analysis Complete
             </motion.div>
             <h1 className="text-2xl font-bold text-white mb-2 text-center">Topps Chrome #23</h1>
             <p className="text-slate-400 text-sm font-medium">2023 • Baseball • Refractor</p>
         </div>

         <div className="grid grid-cols-2 gap-4 mb-8">
             <Card className="border border-white/10 bg-gradient-to-br from-white/5 to-white/0 shadow-lg backdrop-blur-md">
                 <CardContent className="p-5 text-center relative overflow-hidden">
                     <div className="absolute top-0 right-0 p-2 opacity-10">
                        <Sparkles className="w-12 h-12 text-white" />
                     </div>
                     <p className="text-[10px] font-bold text-cyan-400 uppercase tracking-widest mb-1">TruScore</p>
                     <p className="text-4xl font-black text-white tracking-tighter drop-shadow-lg">9.5</p>
                     <p className="text-xs text-slate-400 font-medium mt-1">Gem Mint</p>
                 </CardContent>
             </Card>
             <Card className="border border-white/10 bg-gradient-to-br from-white/5 to-white/0 shadow-lg backdrop-blur-md">
                 <CardContent className="p-5 text-center">
                     <p className="text-[10px] font-bold text-green-400 uppercase tracking-widest mb-1">Market Value</p>
                     <p className="text-4xl font-black text-white tracking-tighter drop-shadow-lg">$185</p>
                     <div className="flex items-center justify-center gap-1 mt-1 text-green-400">
                        <TrendingUp className="w-3 h-3" />
                        <p className="text-xs font-bold">12%</p>
                     </div>
                 </CardContent>
             </Card>
         </div>

         <div className="space-y-8">
             <div>
                 <h3 className="text-sm font-bold text-white uppercase tracking-wider mb-5 pl-1">Sub-Grades</h3>
                 <div className="space-y-4">
                     {[
                         { label: "Centering", score: 9.5, color: "from-cyan-500 to-blue-500" },
                         { label: "Corners", score: 9.0, color: "from-purple-500 to-pink-500" },
                         { label: "Edges", score: 10.0, color: "from-emerald-500 to-teal-500" },
                         { label: "Surface", score: 9.5, color: "from-orange-500 to-red-500" },
                     ].map((item, i) => (
                         <div key={i} className="flex items-center justify-between group">
                             <span className="text-xs font-semibold text-slate-400 w-20 uppercase tracking-wide group-hover:text-white transition-colors">{item.label}</span>
                             <div className="flex-1 h-2 bg-white/5 rounded-full mx-4 overflow-hidden border border-white/5">
                                 <motion.div 
                                    initial={{ width: 0 }}
                                    animate={{ width: `${item.score * 10}%` }}
                                    transition={{ delay: 0.5 + (i * 0.1), duration: 1, type: "spring" }}
                                    className={`h-full rounded-full bg-gradient-to-r ${item.color} shadow-[0_0_10px_rgba(255,255,255,0.3)]`} 
                                 />
                             </div>
                             <span className="text-sm font-bold text-white w-8 text-right font-mono">{item.score}</span>
                         </div>
                     ))}
                 </div>
             </div>

             <Separator className="bg-white/10" />

             <div className="space-y-3">
                 <Button className="w-full h-14 text-base font-bold rounded-xl bg-gradient-to-r from-cyan-600 to-blue-600 hover:from-cyan-500 hover:to-blue-500 border border-white/10 shadow-[0_0_20px_rgba(6,182,212,0.2)]" onClick={onHome}>
                     Submit for Official Grading
                 </Button>
                 <Button variant="outline" className="w-full h-14 text-base font-bold rounded-xl border-white/10 bg-white/5 text-white hover:bg-white/10 hover:text-white backdrop-blur-sm" onClick={onHome}>
                     Save to Collection
                 </Button>
             </div>
             
             <div className="pt-2 flex justify-center">
                 <Button variant="ghost" size="sm" onClick={onHome} className="text-slate-500 hover:text-white transition-colors">
                    <Home className="w-4 h-4 mr-2" />
                    Return to Dashboard
                 </Button>
             </div>
         </div>
      </div>
    </div>
  );
}
