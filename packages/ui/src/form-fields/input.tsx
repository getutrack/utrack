"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../helpers";

const inputVariants = cva(
  "flex w-full rounded-md border bg-transparent px-3 py-2 text-sm ring-offset-background placeholder:text-gray-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 transition-all duration-200",
  {
    variants: {
      variant: {
        default: "border-gray-200 focus-visible:border-primary-500 focus-visible:ring-primary-500",
        error: "border-error-500 focus-visible:border-error-500 focus-visible:ring-error-500",
        success: "border-success-500 focus-visible:border-success-500 focus-visible:ring-success-500",
      },
      size: {
        sm: "h-8 px-3 py-1 text-xs",
        md: "h-10 px-4 py-2 text-sm",
        lg: "h-12 px-4 py-2 text-base",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "md",
    },
  }
);

export interface InputProps
  extends Omit<React.InputHTMLAttributes<HTMLInputElement>, "size">,
    VariantProps<typeof inputVariants> {
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
}

const Input = React.forwardRef<HTMLInputElement, InputProps>(
  ({ className, variant, size, leftIcon, rightIcon, type, ...props }, ref) => {
    return (
      <div className="relative flex items-center">
        {leftIcon && (
          <div className="absolute left-3 flex items-center justify-center text-gray-400">
            {leftIcon}
          </div>
        )}
        <input
          type={type}
          className={cn(
            inputVariants({ variant, size, className }),
            leftIcon && "pl-10",
            rightIcon && "pr-10"
          )}
          ref={ref}
          {...props}
        />
        {rightIcon && (
          <div className="absolute right-3 flex items-center justify-center text-gray-400">
            {rightIcon}
          </div>
        )}
      </div>
    );
  }
);
Input.displayName = "Input";

export { Input };
