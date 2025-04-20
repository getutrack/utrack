export enum ECardVariant {
  WITHOUT_SHADOW = "without-shadow",
  WITH_SHADOW = "with-shadow",
  GRADIENT = "gradient",
  OUTLINED = "outlined",
}
export enum ECardDirection {
  ROW = "row",
  COLUMN = "column",
}
export enum ECardSpacing {
  XS = "xs",
  SM = "sm",
  MD = "md",
  LG = "lg",
  XL = "xl",
}
export type TCardVariant = ECardVariant.WITHOUT_SHADOW | ECardVariant.WITH_SHADOW | ECardVariant.GRADIENT | ECardVariant.OUTLINED;
export type TCardDirection = ECardDirection.ROW | ECardDirection.COLUMN;
export type TCardSpacing = ECardSpacing.XS | ECardSpacing.SM | ECardSpacing.MD | ECardSpacing.LG | ECardSpacing.XL;

export interface ICardProperties {
  [key: string]: string;
}

const DEFAULT_STYLE =
  "bg-custom-background-100 rounded-xl border border-custom-border-200 w-full flex flex-col transition-all duration-200";
export const containerStyle: ICardProperties = {
  [ECardVariant.WITHOUT_SHADOW]: "",
  [ECardVariant.WITH_SHADOW]: "shadow-custom-shadow-sm hover:shadow-custom-shadow-md",
  [ECardVariant.GRADIENT]: "bg-gradient-to-br from-custom-background-100 to-custom-background-90 shadow-custom-shadow-sm hover:shadow-custom-shadow-md",
  [ECardVariant.OUTLINED]: "bg-transparent border-2 hover:border-custom-primary-40 hover:bg-custom-primary-10/30",
};
export const spacings = {
  [ECardSpacing.XS]: "p-2",
  [ECardSpacing.SM]: "p-3",
  [ECardSpacing.MD]: "p-4",
  [ECardSpacing.LG]: "p-5",
  [ECardSpacing.XL]: "p-6",
};
export const directions = {
  [ECardDirection.ROW]: "flex-row items-center gap-3",
  [ECardDirection.COLUMN]: "flex-col gap-3",
};
export const getCardStyle = (variant: TCardVariant, spacing: TCardSpacing, direction: TCardDirection) =>
  `${DEFAULT_STYLE} ${directions[direction]} ${containerStyle[variant]} ${spacings[spacing]}`;
