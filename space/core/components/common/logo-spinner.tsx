"use client";

import { useTheme } from "next-themes";
// ui
import { CircularBarSpinner } from "@utrack/ui";

export const LogoSpinner = () => {
  const { resolvedTheme } = useTheme();
  
  return (
    <div className="h-screen w-full flex min-h-[600px] justify-center items-center">
      <div className="flex items-center justify-center">
        <CircularBarSpinner 
          width="40px" 
          height="40px" 
          className={resolvedTheme === "dark" ? "text-white" : "text-blue-700"}
        />
      </div>
    </div>
  );
};
