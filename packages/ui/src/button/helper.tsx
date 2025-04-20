export type TButtonVariant =
  | "primary"
  | "secondary"
  | "accent-primary"
  | "outline-primary"
  | "neutral-primary"
  | "link-primary"
  | "danger"
  | "success"
  | "warning"
  | "accent-danger"
  | "outline-danger"
  | "outline"
  | "link-danger"
  | "tertiary-danger"
  | "link-neutral"
  | "ghost"
  | "ghost-primary"
  | "link";

export type TButtonSizes = "sm" | "md" | "lg" | "xl";

export interface IButtonStyling {
  [key: string]: {
    default: string;
    hover: string;
    pressed: string;
    disabled: string;
  };
}

enum buttonSizeStyling {
  sm = `px-3 py-1.5 font-medium text-xs rounded-md flex items-center gap-1.5 whitespace-nowrap transition-all duration-200 justify-center`,
  md = `px-4 py-2 font-medium text-sm rounded-md flex items-center gap-1.5 whitespace-nowrap transition-all duration-200 justify-center`,
  lg = `px-5 py-2.5 font-medium text-sm rounded-md flex items-center gap-1.5 whitespace-nowrap transition-all duration-200 justify-center`,
  xl = `px-6 py-3.5 font-medium text-sm rounded-md flex items-center gap-1.5 whitespace-nowrap transition-all duration-200 justify-center`,
}

enum buttonIconStyling {
  sm = "h-3.5 w-3.5 flex justify-center items-center overflow-hidden my-0.5 flex-shrink-0",
  md = "h-4 w-4 flex justify-center items-center overflow-hidden my-0.5 flex-shrink-0",
  lg = "h-4.5 w-4.5 flex justify-center items-center overflow-hidden my-0.5 flex-shrink-0",
  xl = "h-5 w-5 flex justify-center items-center overflow-hidden my-0.5 flex-shrink-0",
}

export const buttonStyling: IButtonStyling = {
  primary: {
    default: `text-white bg-custom-primary-60 shadow-sm`,
    hover: `hover:bg-custom-primary-70 hover:shadow-md`,
    pressed: `focus:ring-2 focus:ring-custom-primary-40 focus:ring-offset-1 active:bg-custom-primary-80`,
    disabled: `cursor-not-allowed !bg-custom-primary-40 hover:bg-custom-primary-40 hover:shadow-none opacity-70`,
  },
  "accent-primary": {
    default: `bg-custom-primary-10 text-custom-primary-70 shadow-sm`,
    hover: `hover:bg-custom-primary-20 hover:text-custom-primary-80`,
    pressed: `focus:ring-2 focus:ring-custom-primary-30 focus:ring-offset-1 active:bg-custom-primary-30`,
    disabled: `cursor-not-allowed !text-custom-primary-40 opacity-70`,
  },
  "outline-primary": {
    default: `text-custom-primary-70 bg-transparent border border-custom-primary-40 shadow-sm`,
    hover: `hover:bg-custom-primary-10 hover:border-custom-primary-60`,
    pressed: `focus:ring-2 focus:ring-custom-primary-30 focus:ring-offset-1 active:bg-custom-primary-20`,
    disabled: `cursor-not-allowed !text-custom-primary-40 !border-custom-primary-30 opacity-70`,
  },
  "neutral-primary": {
    default: `text-custom-text-200 bg-white border border-custom-border-300 shadow-sm`,
    hover: `hover:bg-custom-background-90 hover:text-custom-text-100`,
    pressed: `focus:ring-2 focus:ring-custom-border-300 focus:ring-offset-1 active:bg-custom-background-80`,
    disabled: `cursor-not-allowed !text-custom-text-400 opacity-70`,
  },
  "link-primary": {
    default: `text-custom-primary-70 bg-transparent`,
    hover: `hover:text-custom-primary-80 hover:underline`,
    pressed: `focus:text-custom-primary-90 active:text-custom-primary-90`,
    disabled: `cursor-not-allowed !text-custom-primary-40 opacity-70`,
  },

  danger: {
    default: `text-white bg-red-500 shadow-sm`,
    hover: `hover:bg-red-600 hover:shadow-md`,
    pressed: `focus:ring-2 focus:ring-red-300 focus:ring-offset-1 active:bg-red-700`,
    disabled: `cursor-not-allowed !bg-red-300 hover:shadow-none opacity-70`,
  },
  "accent-danger": {
    default: `text-red-600 bg-red-50 shadow-sm`,
    hover: `hover:text-red-700 hover:bg-red-100`,
    pressed: `focus:ring-2 focus:ring-red-200 focus:ring-offset-1 active:bg-red-200`,
    disabled: `cursor-not-allowed !text-red-300 opacity-70`,
  },
  "outline-danger": {
    default: `text-red-500 bg-transparent border border-red-300 shadow-sm`,
    hover: `hover:bg-red-50 hover:border-red-500`,
    pressed: `focus:ring-2 focus:ring-red-200 focus:ring-offset-1 active:bg-red-100`,
    disabled: `cursor-not-allowed !text-red-300 !border-red-200 opacity-70`,
  },
  "link-danger": {
    default: `text-red-500 bg-transparent`,
    hover: `hover:text-red-600 hover:underline`,
    pressed: `focus:text-red-700 active:text-red-700`,
    disabled: `cursor-not-allowed !text-red-300 opacity-70`,
  },
  "tertiary-danger": {
    default: `text-red-500 bg-white border border-red-200 shadow-sm`,
    hover: `hover:bg-red-50 hover:border-red-300`,
    pressed: `focus:ring-2 focus:ring-red-200 focus:ring-offset-1 active:bg-red-100`,
    disabled: `cursor-not-allowed !text-red-300 opacity-70`,
  },
  "link-neutral": {
    default: `text-custom-text-300 bg-transparent`,
    hover: `hover:text-custom-text-200 hover:underline`,
    pressed: `focus:text-custom-text-100 active:text-custom-text-100`,
    disabled: `cursor-not-allowed !text-custom-text-400 opacity-70`,
  },
};

export const getButtonStyling = (variant: TButtonVariant, size: TButtonSizes, disabled: boolean = false): string => {
  let tempVariant: string = ``;
  const currentVariant = buttonStyling[variant];

  tempVariant = `${currentVariant.default} ${disabled ? currentVariant.disabled : currentVariant.hover} ${
    currentVariant.pressed
  }`;

  let tempSize: string = ``;
  if (size) tempSize = buttonSizeStyling[size];
  return `${tempVariant} ${tempSize}`;
};

export const getIconStyling = (size: TButtonSizes): string => {
  let icon: string = ``;
  if (size) icon = buttonIconStyling[size];
  return icon;
};
