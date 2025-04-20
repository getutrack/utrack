"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

// helpers
import { cn } from "../helpers";
import { TButtonVariant } from "./helper";

export const buttonVariants = cva(
  "relative inline-flex items-center justify-center whitespace-nowrap rounded-md text-sm font-medium transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-70",
  {
    variants: {
      variant: {
        primary: "bg-primary-600 text-white hover:bg-primary-700 active:bg-primary-800 focus-visible:ring-primary-500",
        secondary: "bg-gray-100 text-gray-900 hover:bg-gray-200 active:bg-gray-300 focus-visible:ring-gray-500",
        danger: "bg-error-500 text-white hover:bg-error-600 active:bg-error-700 focus-visible:ring-error-500",
        success: "bg-success-500 text-white hover:bg-success-600 active:bg-success-700 focus-visible:ring-success-500",
        warning: "bg-warning-500 text-white hover:bg-warning-600 active:bg-warning-700 focus-visible:ring-warning-500",
        outline: "border border-gray-300 bg-transparent text-gray-700 hover:bg-gray-50 active:bg-gray-100 focus-visible:ring-gray-500",
        "outline-primary": "border border-primary-500 bg-transparent text-primary-600 hover:bg-primary-50 active:bg-primary-100 focus-visible:ring-primary-500",
        "outline-danger": "border border-error-500 bg-transparent text-error-600 hover:bg-error-50 active:bg-error-100 focus-visible:ring-error-500",
        ghost: "bg-transparent text-gray-700 hover:bg-gray-100 active:bg-gray-200 focus-visible:ring-gray-500",
        "ghost-primary": "bg-transparent text-primary-600 hover:bg-primary-50 active:bg-primary-100 focus-visible:ring-primary-500",
        link: "bg-transparent text-primary-600 underline-offset-4 hover:underline focus-visible:ring-primary-500",
        "link-primary": "bg-transparent text-primary-600 underline-offset-4 hover:underline focus-visible:ring-primary-500",
        "link-danger": "bg-transparent text-error-600 underline-offset-4 hover:underline focus-visible:ring-error-500",
        "link-neutral": "bg-transparent text-gray-600 underline-offset-4 hover:underline focus-visible:ring-gray-500",
        "neutral-primary": "bg-gray-50 text-gray-700 hover:bg-gray-100 active:bg-gray-200 focus-visible:ring-gray-500",
        "accent-primary": "bg-primary-50 text-primary-700 hover:bg-primary-100 active:bg-primary-200 focus-visible:ring-primary-500",
        "accent-danger": "bg-error-50 text-error-700 hover:bg-error-100 active:bg-error-200 focus-visible:ring-error-500",
        "tertiary-danger": "bg-white border border-error-200 text-error-500 hover:bg-error-50 hover:border-error-300 active:bg-error-100 focus-visible:ring-error-200",
      },
      size: {
        xs: "h-7 px-2.5 text-xs",
        sm: "h-8 px-3 text-sm",
        md: "h-9 px-4 text-sm",
        lg: "h-10 px-5 text-base",
        xl: "h-12 px-6 text-lg",
      },
      rounded: {
        default: "rounded-md",
        full: "rounded-full",
      },
      fontWeight: {
        medium: "font-medium",
        semibold: "font-semibold",
        bold: "font-bold",
      },
      width: {
        auto: "w-auto",
        full: "w-full",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "md",
      rounded: "default",
      fontWeight: "medium",
      width: "auto",
    },
  }
);

export interface ButtonProps
  extends Omit<React.ButtonHTMLAttributes<HTMLButtonElement>, "disabled">,
    Omit<VariantProps<typeof buttonVariants>, "variant"> {
  loading?: boolean;
  disabled?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  children: React.ReactNode;
  variant?: TButtonVariant;
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant,
      size,
      rounded,
      fontWeight,
      width,
      loading = false,
      disabled = false,
      leftIcon,
      rightIcon,
      children,
      ...props
    },
    ref
  ) => {
    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        disabled={isDisabled}
        className={cn(
          buttonVariants({
            variant,
            size,
            rounded,
            fontWeight,
            width,
            className,
          }),
          loading && "relative text-transparent transition-none hover:text-transparent",
          className
        )}
        {...props}
      >
        {leftIcon && <span className={cn("mr-2", loading && "invisible")}>{leftIcon}</span>}
        <span className={loading ? "invisible" : ""}>{children}</span>
        {rightIcon && <span className={cn("ml-2", loading && "invisible")}>{rightIcon}</span>}
        {loading && (
          <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
            <svg
              className="animate-spin h-4 w-4"
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
            >
              <circle
                className="opacity-25"
                cx="12"
                cy="12"
                r="10"
                stroke="currentColor"
                strokeWidth="4"
              ></circle>
              <path
                className="opacity-75"
                fill="currentColor"
                d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
              ></path>
            </svg>
          </div>
        )}
      </button>
    );
  }
);

Button.displayName = "Button";

export { Button };
