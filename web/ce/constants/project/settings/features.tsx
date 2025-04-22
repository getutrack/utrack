import { ReactNode } from "react";
import { FileText, Layers, Timer } from "lucide-react";
import { IProject } from "@utrack/types";
import { ContrastIcon, DiceIcon, Intake } from "@utrack/ui";
import {
  CalendarClock,
  BrainCircuit,
  Wrench,
  LayoutGrid,
  Presentation,
  Zap,
  Webhook,
  EyeIcon,
  TrendingUpIcon,
  SprayCan,
  ClipboardList,
} from "lucide-react";
import { IProjectFeature } from "@/types";

export type TProperties = {
  property: string;
  title: string;
  description: string;
  icon: ReactNode;
  isPro: boolean;
  isEnabled: boolean;
  renderChildren?: (
    currentProjectDetails: IProject,
    isAdmin: boolean,
    handleSubmit: (featureKey: string, featureProperty: string) => Promise<void>
  ) => ReactNode;
};
export type TFeatureList = {
  [key: string]: TProperties;
};

export type TProjectFeatures = {
  [key: string]: {
    title: string;
    description: string;
    featureList: TFeatureList;
  };
};

export const PROJECT_FEATURES_LIST: IProjectFeature[] = [
  {
    title: "AI",
    description: "Generate description, summary, acceptance criteria and subtasks for your issues using AI",
    key: "ai_enabled",
    icon: BrainCircuit,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "Cycles",
    description: "Group issues together into delivery cycles",
    key: "cycle_enabled",
    icon: CalendarClock,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "Cycles automation",
    description: "Automatically create cycles, drag incomplete issues and send reports",
    key: "cycle_automation_enabled",
    icon: Wrench,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "Tasks in views",
    description: "Add tasks to different views",
    key: "task_views_enabled",
    icon: LayoutGrid,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "Modules",
    description: "Track larger collections of issues across projects",
    key: "module_enabled",
    icon: Layers,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "Boards",
    description: "Track bigger pieces of work",
    key: "board_enabled",
    icon: ClipboardList,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "Views",
    description: "Customize your project view, save filters and layouts",
    key: "issue_views_enabled",
    icon: EyeIcon,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "Custom roles",
    description: "Control authorization through predefined roles",
    key: "project_custom_roles_enabled",
    icon: Presentation,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "SLA",
    description: "Track the performance of your team using SLA rules",
    key: "sla_enabled",
    icon: TrendingUpIcon,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "Automations",
    description: "Create automation based on the issue event, a condition and an action",
    key: "automation_enabled",
    icon: Zap,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "Webhooks",
    description: "Create endpoints for information to be sent to when certain events are triggered",
    key: "webhook_enabled",
    icon: Webhook,
    isAvailable: true,
    isPro: false,
  },
  {
    title: "API",
    description: "Enable API access to your project",
    key: "api_enabled",
    icon: SprayCan,
    isAvailable: true,
    isPro: false,
  },
];
