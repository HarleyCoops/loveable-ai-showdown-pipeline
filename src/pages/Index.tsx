
import React from "react";

const Index = () => {
  return (
    <div className="relative min-h-screen bg-background flex flex-col">
      {/* Map background layer with less blur and higher visibility */}
      <img
        src="/lovable-uploads/cab9a7dc-b686-4496-854d-dfe6f4b8d09e.png"
        alt="Historic First Nations Map"
        className="absolute inset-0 w-full h-full object-cover opacity-80 blur-[0.5px] pointer-events-none select-none z-0"
        aria-hidden="true"
      />
      {/* Lighter overlay for background */}
      <div className="absolute inset-0 bg-background/30 z-0" aria-hidden="true" />

      {/* Header at the top for the title and subtitle */}
      <header className="relative z-10 w-full px-4 pt-8 pb-4 flex flex-col items-center backdrop-blur-sm bg-background/70">
        <h1 className="text-2xl md:text-4xl font-bold text-foreground drop-shadow mb-2 text-center">
          Welcome to Dictionary to OpenAI: A Low-Resource Language Trainer
        </h1>
        <p className="text-base md:text-xl text-muted-foreground font-medium text-center">
          Pardon the landing page while I build in the background for a few more hours
        </p>
      </header>

      {/* Content area without video */}
      <main className="flex-1 flex items-center justify-center relative z-10">
        <div className="w-full max-w-xl px-4">
          <div className="aspect-video bg-muted rounded-xl flex items-center justify-center shadow-lg border-2 border-muted-foreground/30">
            {/* Video removed as requested */}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Index;
