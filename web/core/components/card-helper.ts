/**
 * Local copies of card helper enums
 */

export enum ECardVariant {
  "without-shadow" = "without-shadow",
  "with-shadow" = "with-shadow",
}

export type TCardVariant = `${ECardVariant}`;

export enum ECardDirection {
  "row" = "row",
  "column" = "column",
}

export type TCardDirection = `${ECardDirection}`;

export enum ECardSpacing {
  "XS" = "XS",
  "SM" = "SM",
  "MD" = "MD",
  "LG" = "LG",
  "XL" = "XL",
}

export type TCardSpacing = `${ECardSpacing}`; 