import { Github, MessageCircle, Heart } from "lucide-react";

const Footer = () => {
  return (
    <footer className="border-t border-neon-blue/20 bg-gradient-to-b from-transparent to-muted/10 py-12 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 text-muted-foreground mb-4">
            <span className="text-accent font-semibold">Open Source</span>
            <span>•</span>
            <span className="text-neon-blue font-semibold">MIT License</span>
            <span>•</span>
            <span>Built with</span>
            <Heart className="w-4 h-4 text-neon-pink inline mx-1 animate-pulse" />
          </div>
          
          <p className="text-sm text-muted-foreground mb-6">
            Built on <span className="text-foreground font-mono">FastAPI</span> +{" "}
            <span className="text-foreground font-mono">vLLM</span> +{" "}
            <span className="text-foreground font-mono">F5-TTS</span>
          </p>
        </div>

        {/* Social Links */}
        <div className="flex justify-center gap-6">
          <a
            href="#"
            className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors group"
          >
            <div className="w-10 h-10 rounded-full bg-muted/30 border border-neon-blue/30 flex items-center justify-center group-hover:border-neon-blue group-hover:bg-neon-blue/10 transition-all">
              <Github className="w-5 h-5" />
            </div>
            <span className="text-sm font-medium">GitHub</span>
          </a>
          
          <a
            href="#"
            className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors group"
          >
            <div className="w-10 h-10 rounded-full bg-muted/30 border border-neon-pink/30 flex items-center justify-center group-hover:border-neon-pink group-hover:bg-neon-pink/10 transition-all">
              <MessageCircle className="w-5 h-5" />
            </div>
            <span className="text-sm font-medium">Discord</span>
          </a>
          
          <a
            href="#"
            className="flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors group"
          >
            <div className="w-10 h-10 rounded-full bg-muted/30 border border-accent/30 flex items-center justify-center group-hover:border-accent group-hover:bg-accent/10 transition-all">
              <Heart className="w-5 h-5" />
            </div>
            <span className="text-sm font-medium">HuggingFace</span>
          </a>
        </div>

        <div className="text-center mt-8 text-xs text-muted-foreground">
          <p>© 2024 AVATAR Project. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
