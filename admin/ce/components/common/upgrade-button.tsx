"use client";

import React from "react";
// icons
import { SquareArrowOutUpRight } from "lucide-react";
// ui
import { getButtonStyling } from "@utrack/ui";
// helpers
import { cn } from "@/helpers/common.helper";

export const UpgradeButton: React.FC = () => (
  <a href="https://getutrack.io/one" target="_blank" className={cn(getButtonStyling("primary", "sm"))}>
    Available on One
    <SquareArrowOutUpRight className="h-3.5 w-3.5 p-0.5" />
  </a>
);
