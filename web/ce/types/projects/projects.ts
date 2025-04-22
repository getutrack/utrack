import { IProject } from "@utrack/types";
import { ReactNode } from "react";

export type TProject = IProject;

export interface IProjectFeature {
  title: string;
  description: string;
  key: string;
  icon: React.ElementType;
  isAvailable: boolean;
  isPro: boolean;
}
