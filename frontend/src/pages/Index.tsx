import { useState, useRef } from "react";
import Hero from "@/components/Hero";
import DemoPanel from "@/components/DemoPanel";
import VoiceProfileManager from "@/components/VoiceProfileManager";
import ConversationHistory from "@/components/ConversationHistory";
import Footer from "@/components/Footer";

const Index = () => {
  const demoPanelRef = useRef<HTMLDivElement>(null);

  const scrollToDemo = () => {
    demoPanelRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Hero onStartDemo={scrollToDemo} />
      <div ref={demoPanelRef}>
        <DemoPanel />
      </div>
      <VoiceProfileManager />
      <ConversationHistory />
      <Footer />
    </div>
  );
};

export default Index;
