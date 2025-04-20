import React from "react";
import { cn } from "../../helpers";

type HeadingLevel = "h1" | "h2" | "h3" | "h4" | "h5" | "h6";
type HeadingSize = "xs" | "sm" | "md" | "lg" | "xl" | "2xl" | "3xl" | "4xl";
type HeadingIntent = "default" | "primary" | "secondary" | "muted";

type Props = {
  children: React.ReactNode;
  as?: HeadingLevel;
  size?: HeadingSize;
  intent?: HeadingIntent;
  className?: string;
  noMargin?: boolean;
};

const Heading = ({
  children,
  as: Component = "h2",
  size = "md",
  intent = "default",
  className,
  noMargin = false,
}: Props) => {
  const sizeStyles: Record<HeadingSize, string> = {
    xs: "text-base font-semibold leading-tight",
    sm: "text-lg font-semibold leading-tight",
    md: "text-xl font-semibold leading-tight",
    lg: "text-2xl font-semibold leading-tight",
    xl: "text-3xl font-bold leading-tight tracking-tight",
    "2xl": "text-4xl font-bold leading-tight tracking-tight",
    "3xl": "text-5xl font-bold leading-tight tracking-tight",
    "4xl": "text-6xl font-bold leading-tight tracking-tight",
  };

  const intentStyles: Record<HeadingIntent, string> = {
    default: "text-custom-text-100",
    primary: "text-custom-primary-80",
    secondary: "text-custom-text-200",
    muted: "text-custom-text-300",
  };

  return (
    <Component
      className={cn(
        sizeStyles[size],
        intentStyles[intent],
        !noMargin && "mb-4", 
        className
      )}
    >
      {children}
    </Component>
  );
};

export { Heading }; 