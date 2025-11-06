import { Button } from "@/components/ui/button";
import { Mic } from "lucide-react";

const Hero = ({ onStartDemo }: { onStartDemo: () => void }) => {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
      {/* Ambient Background Effects */}
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-neon-blue/5 to-transparent" />
      <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-neon-blue/10 rounded-full blur-3xl animate-float" />
      <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-neon-pink/10 rounded-full blur-3xl animate-float" style={{ animationDelay: "1s" }} />
      
      <div className="relative z-10 text-center px-4 max-w-5xl mx-auto">
        {/* Animated Pulse Ring */}
        <div className="mb-12 flex justify-center">
          <div className="relative w-64 h-64 md:w-80 md:h-80">
            {/* Outer ring - blue glow */}
            <div className="absolute inset-0 rounded-full border-4 border-neon-blue animate-pulse-ring" 
                 style={{ filter: "drop-shadow(0 0 30px hsl(var(--neon-blue) / 0.6))" }} />
            
            {/* Middle ring - pink/magenta */}
            <div className="absolute inset-4 rounded-full border-4 border-neon-pink animate-pulse-ring border-dashed" 
                 style={{ animationDelay: "0.5s", filter: "drop-shadow(0 0 25px hsl(var(--neon-pink) / 0.5))" }} />
            
            {/* Inner glow */}
            <div className="absolute inset-8 rounded-full bg-gradient-to-br from-neon-blue/20 via-neon-pink/20 to-transparent blur-xl" />
            
            {/* Center dot pattern */}
            <div className="absolute bottom-12 right-12 flex gap-1">
              {[...Array(8)].map((_, i) => (
                <div
                  key={i}
                  className="w-1 h-8 bg-accent rounded-full animate-pulse"
                  style={{ 
                    animationDelay: `${i * 0.1}s`,
                    height: `${Math.random() * 32 + 8}px`,
                    filter: "drop-shadow(0 0 8px hsl(var(--gold) / 0.8))"
                  }}
                />
              ))}
            </div>
            
            {/* Mic icon in center */}
            <div className="absolute inset-0 flex items-center justify-center">
              <Mic className="w-12 h-12 text-accent animate-glow-pulse" />
            </div>
          </div>
        </div>

        {/* Hero Text */}
        <h1 className="text-5xl md:text-7xl font-bold mb-6 bg-neon-gradient bg-clip-text text-transparent animate-shimmer bg-[length:200%_100%]">
          Local AI Voice Agents
        </h1>
        
        <p className="text-xl md:text-2xl text-muted-foreground mb-12 max-w-2xl mx-auto">
          Try voice-to-voice AI chat with{" "}
          <span className="text-accent font-semibold">no cloud connection</span>.
        </p>

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <Button 
            onClick={onStartDemo}
            size="lg"
            className="bg-neon-gradient hover:opacity-90 text-white font-semibold px-8 py-6 text-lg rounded-xl shadow-lg hover:shadow-neon-blue/50 transition-all duration-300"
          >
            <Mic className="w-5 h-5 mr-2" />
            Start Voice Demo
          </Button>
          
          <Button
            variant="outline"
            size="lg"
            className="border-2 border-neon-blue/50 bg-transparent hover:bg-neon-blue/10 text-foreground font-semibold px-8 py-6 text-lg rounded-xl backdrop-blur-sm transition-all duration-300"
          >
            View Security Architecture
          </Button>
        </div>
      </div>
    </section>
  );
};

export default Hero;
