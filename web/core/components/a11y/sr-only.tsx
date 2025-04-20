"use client";

import React from "react";
import { cn } from "@/helpers/common.helper";

interface SrOnlyProps {
  children: React.ReactNode;
  className?: string;
}

export const SrOnly = ({ children, className }: SrOnlyProps) => {
  return (
    <span
      className={cn(
        "absolute w-px h-px p-0 -m-px overflow-hidden whitespace-nowrap border-0",
        "clip-path: inset(50%)",
        className
      )}
    >
      {children}
    </span>
  );
}; 