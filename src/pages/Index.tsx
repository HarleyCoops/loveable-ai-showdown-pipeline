
const Index = () => {
  return (
    <div className="relative min-h-screen flex items-center justify-center bg-background overflow-hidden">
      {/* Map background layer */}
      <img
        src="/lovable-uploads/cab9a7dc-b686-4496-854d-dfe6f4b8d09e.png"
        alt="Historic First Nations Map"
        className="absolute inset-0 w-full h-full object-cover opacity-60 blur-[2px] pointer-events-none select-none"
        aria-hidden="true"
      />
      {/* Overlay for subtle dimming if needed */}
      <div className="absolute inset-0 bg-background/70" aria-hidden="true" />
      {/* Foreground content */}
      <div className="relative text-center py-24 px-6 z-10">
        <h1 className="text-3xl md:text-5xl font-bold mb-6 text-foreground drop-shadow">
          Welcome to Dictionary to OpenAI: A Low-Resource Language Trainer
        </h1>
        <p className="text-lg md:text-2xl text-muted-foreground font-medium">
          pardon us while we build in the background for a few more hours
        </p>
      </div>
    </div>
  );
};

export default Index;
