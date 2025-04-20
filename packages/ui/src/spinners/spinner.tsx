"use client";

import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "../../helpers";

const spinnerVariants = cva(
  "animate-spin rounded-full border-solid border-t-transparent",
  {
    variants: {
      variant: {
        primary: "border-primary-600 border-t-transparent",
        secondary: "border-gray-600 border-t-transparent",
        white: "border-white border-t-transparent",
      },
      size: {
        xs: "h-3 w-3 border-[2px]",
        sm: "h-4 w-4 border-[2px]",
        md: "h-6 w-6 border-2",
        lg: "h-8 w-8 border-[3px]",
        xl: "h-12 w-12 border-4",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "md",
    },
  }
);

export interface SpinnerProps extends React.HTMLAttributes<HTMLDivElement>, VariantProps<typeof spinnerVariants> {
  label?: string;
}

const Spinner = React.forwardRef<HTMLDivElement, SpinnerProps>(
  ({ className, variant, size, label, ...props }, ref) => {
    return (
      <div ref={ref} className="flex flex-col items-center justify-center" {...props}>
        <div className={cn(spinnerVariants({ variant, size, className }))} aria-hidden="true" />
        {label && <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">{label}</p>}
      </div>
    );
  }
);

Spinner.displayName = "Spinner";

export { Spinner }; 