
import React from 'react';

export function MobileWrapper({ children }: { children: React.ReactNode }) {
  return (
    <div className="w-full max-w-md h-[800px] bg-slate-950 rounded-[3rem] shadow-2xl overflow-hidden relative border-[8px] border-slate-900 flex flex-col ring-1 ring-white/10">
      {/* Abstract Background Gradient */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-violet-600/20 rounded-full blur-[80px]"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-cyan-600/20 rounded-full blur-[80px]"></div>
        <div className="absolute top-[40%] left-[30%] w-[40%] h-[40%] bg-blue-600/10 rounded-full blur-[60px]"></div>
      </div>

      {/* Status Bar Mock */}
      <div className="h-12 flex justify-between items-center px-6 pt-2 shrink-0 z-50 text-white">
        <span className="font-semibold text-sm">9:41</span>
        <div className="w-20 h-6 bg-black/50 rounded-full absolute left-1/2 -translate-x-1/2 top-3 backdrop-blur-md border border-white/5"></div>
        <div className="flex gap-1.5">
          <div className="w-4 h-2.5 bg-white rounded-sm"></div>
          <div className="w-4 h-2.5 bg-white rounded-sm"></div>
          <div className="w-5 h-2.5 border border-white rounded-sm bg-white"></div>
        </div>
      </div>
      
      <div className="flex-1 flex flex-col relative overflow-y-auto overflow-x-hidden no-scrollbar z-40">
        {children}
      </div>

      {/* Home Indicator */}
      <div className="h-6 shrink-0 flex justify-center items-center pb-2 pt-2 z-50">
        <div className="w-32 h-1 bg-white/20 rounded-full backdrop-blur-md"></div>
      </div>
    </div>
  );
}
