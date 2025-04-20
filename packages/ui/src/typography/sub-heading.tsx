import React from "react";
import { cn } from "../../helpers";

type Props = {
  children: React.ReactNode;
  className?: string;
  noMargin?: boolean;
  intent?: "primary" | "secondary" | "default";
  size?: "sm" | "md" | "lg";
};

const SubHeading = ({ 
  children, 
  className, 
  noMargin, 
  intent = "default",
  size = "md"
}: Props) => {
  const intentStyles = {
    default: "text-custom-text-200",
    primary: "text-custom-primary-70",
    secondary: "text-custom-text-300",
  };

  const sizeStyles = {
    sm: "text-lg font-medium leading-6",
    md: "text-xl font-medium leading-7",
    lg: "text-2xl font-medium leading-8",
  };

  return (
    <h3 
      className={cn(
        intentStyles[intent],
        sizeStyles[size],
        !noMargin && "mb-3",
        className
      )}
    >
      {children}
    </h3>
  );
};

export { SubHeading };
