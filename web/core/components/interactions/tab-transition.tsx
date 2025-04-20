"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/helpers/common.helper";

interface Tab {
  id: string;
  label: string;
}

interface AnimatedTabsProps {
  tabs: Tab[];
  defaultTab?: string;
  onChange?: (tabId: string) => void;
  className?: string;
}

export const AnimatedTabs = ({ tabs, defaultTab, onChange, className }: AnimatedTabsProps) => {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
    onChange?.(tabId);
  };

  return (
    <div className={cn("relative flex rounded-lg p-1 bg-gray-100 dark:bg-gray-800", className)}>
      {tabs.map((tab) => (
        <button
          key={tab.id}
          onClick={() => handleTabChange(tab.id)}
          className={cn(
            "relative z-10 px-4 py-2 text-sm font-medium transition-colors rounded-md",
            activeTab === tab.id
              ? "text-gray-900 dark:text-white"
              : "text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
          )}
        >
          {tab.label}
        </button>
      ))}
      
      {/* Animated highlight */}
      <motion.div
        className="absolute z-0 rounded-md bg-white dark:bg-gray-700 shadow-sm transition-colors"
        layoutId="tab-highlight"
        transition={{ type: "spring", stiffness: 500, damping: 30 }}
        style={{
          width: `${100 / tabs.length}%`,
          height: "80%",
          top: "10%",
          left: `${(tabs.findIndex((tab) => tab.id === activeTab) * 100) / tabs.length}%`,
        }}
      />
    </div>
  );
}; 