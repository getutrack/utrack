"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/helpers/common.helper";

interface ButtonHoverEffectProps {
  children: React.ReactNode;
  className?: string;
}

export const ButtonHoverEffect = ({ children, className }: ButtonHoverEffectProps) => {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <motion.div
      className={cn("relative inline-block", className)}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
    >
      {children}
      {isHovered && (
        <motion.div
          layoutId="buttonHoverEffect"
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.85 }}
          transition={{ type: "spring", stiffness: 300, damping: 30 }}
          className="absolute inset-0 bg-primary-100/20 dark:bg-primary-900/30 rounded-md -z-10"
        />
      )}
    </motion.div>
  );
}; 