"use client";
import React, { FC } from "react";
// helpers
import { cn } from "@/helpers/common.helper";

type TSidebarNavItem = {
  className?: string;
  isActive?: boolean;
  children?: React.ReactNode;
  icon?: React.ReactNode;
  onClick?: () => void;
};

export const SidebarNavItem: FC<TSidebarNavItem> = (props) => {
  const { className, isActive, children, icon, onClick } = props;
  return (
    <div
      className={cn(
        "cursor-pointer relative group w-full flex items-center justify-between gap-2 rounded-md px-3 py-2.5 outline-none transition-all duration-200",
        {
          "bg-custom-primary-10 text-custom-primary-70 font-medium shadow-sm": isActive,
          "text-custom-sidebar-text-200 hover:bg-custom-sidebar-background-90 hover:text-custom-text-100 active:bg-custom-sidebar-background-80":
            !isActive,
        },
        className
      )}
      onClick={onClick}
    >
      {/* Highlight bar for active state */}
      {isActive && (
        <div className="absolute left-0 top-0 bottom-0 w-1 bg-custom-primary-70 rounded-l-md" />
      )}
      
      {/* Icon with positioning */}
      {icon && (
        <div className="flex-shrink-0 w-5 h-5 flex items-center justify-center">
          {icon}
        </div>
      )}
      
      {/* Content with flex layout */}
      <div className="flex-grow flex items-center justify-between">
        {children}
      </div>
    </div>
  );
};
