import React from "react";
import { cn } from "../../helpers";

type TextElement = "p" | "span" | "div" | "label";
type TextSize = "xs" | "sm" | "md" | "lg" | "xl";
type TextWeight = "normal" | "medium" | "semibold" | "bold";
type TextIntent = "default" | "primary" | "secondary" | "muted" | "error" | "success";

interface Props {
  children: React.ReactNode;
  as?: TextElement;
  size?: TextSize;
  weight?: TextWeight;
  intent?: TextIntent;
  className?: string;
  truncate?: boolean;
  noMargin?: boolean;
}

const Text = ({
  children,
  as: Component = "p",
  size = "md",
  weight = "normal",
  intent = "default",
  className = "",
  truncate = false,
  noMargin = false,
}: Props) => {
  const sizeStyles: Record<TextSize, string> = {
    xs: "text-xs",
    sm: "text-sm",
    md: "text-base",
    lg: "text-lg",
    xl: "text-xl",
  };

  const weightStyles: Record<TextWeight, string> = {
    normal: "font-normal",
    medium: "font-medium",
    semibold: "font-semibold",
    bold: "font-bold",
  };

  const intentStyles: Record<TextIntent, string> = {
    default: "text-custom-text-200",
    primary: "text-custom-text-100",
    secondary: "text-custom-text-300",
    muted: "text-custom-text-400",
    error: "text-custom-error-200",
    success: "text-green-600",
  };

  return (
    <Component
      className={cn(
        sizeStyles[size],
        weightStyles[weight],
        intentStyles[intent],
        !noMargin && "mb-2",
        truncate && "truncate",
        className
      )}
    >
      {children}
    </Component>
  );
};

export { Text }; 