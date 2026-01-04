
import React, { useState } from 'react';
import { MobileWrapper } from './components/layout/MobileWrapper';
import { HomeScreen } from './components/screens/HomeScreen';
import { CameraScreen } from './components/screens/CameraScreen';
import { AnalyzingScreen } from './components/screens/AnalyzingScreen';
import { ResultsScreen } from './components/screens/ResultsScreen';
import { Toaster } from 'sonner';

export type Screen = 'home' | 'camera-front' | 'camera-back' | 'analyzing' | 'results';

export default function App() {
  const [currentScreen, setCurrentScreen] = useState<Screen>('home');
  const [frontImage, setFrontImage] = useState<string | null>(null);
  const [backImage, setBackImage] = useState<string | null>(null);

  const resetScan = () => {
    setFrontImage(null);
    setBackImage(null);
    setCurrentScreen('home');
  };

  const handleCapture = (image: string, side: 'front' | 'back') => {
    if (side === 'front') {
      setFrontImage(image);
    } else {
      setBackImage(image);
    }
    setCurrentScreen('home');
  };

  const startAnalysis = () => {
    setCurrentScreen('analyzing');
  };

  return (
    <div className="min-h-screen bg-neutral-950 flex justify-center items-center p-4 bg-[url('https://images.unsplash.com/photo-1534068590799-09895a701e3e?q=80&w=2500&auto=format&fit=crop')] bg-cover bg-center bg-no-repeat bg-blend-overlay bg-fixed">
       <div className="absolute inset-0 bg-neutral-950/80 backdrop-blur-sm pointer-events-none"></div>
      <MobileWrapper>
        {currentScreen === 'home' && (
          <HomeScreen 
            frontImage={frontImage}
            backImage={backImage}
            onScanFront={() => setCurrentScreen('camera-front')}
            onScanBack={() => setCurrentScreen('camera-back')}
            onAnalyze={startAnalysis}
          />
        )}
        
        {currentScreen === 'camera-front' && (
          <CameraScreen 
            side="front"
            onCapture={(img) => handleCapture(img, 'front')}
            onBack={() => setCurrentScreen('home')}
          />
        )}

        {currentScreen === 'camera-back' && (
          <CameraScreen 
            side="back"
            onCapture={(img) => handleCapture(img, 'back')}
            onBack={() => setCurrentScreen('home')}
          />
        )}

        {currentScreen === 'analyzing' && (
          <AnalyzingScreen 
            onComplete={() => setCurrentScreen('results')}
          />
        )}

        {currentScreen === 'results' && (
          <ResultsScreen 
            frontImage={frontImage}
            onHome={resetScan}
          />
        )}
      </MobileWrapper>
      <Toaster />
    </div>
  );
}
