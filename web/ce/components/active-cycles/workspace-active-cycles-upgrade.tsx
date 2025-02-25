"use client";

import React from "react";
import { observer } from "mobx-react";
import Image from "next/image";
// ui
import { ContentWrapper, getButtonStyling } from "@utrack/ui";
// components
import { ProIcon } from "@/components/common";
// constants
import { MARKETING_PRICING_PAGE_LINK } from "@/constants/common";
import { WORKSPACE_ACTIVE_CYCLES_DETAILS } from "@/constants/cycle";
// helper
import { cn } from "@/helpers/common.helper";
// hooks
import { useUser } from "@/hooks/store";

export const WorkspaceActiveCyclesUpgrade = observer(() => {
  // store hooks
  const {
    userProfile: { data: userProfile },
  } = useUser();

  const isDarkMode = userProfile?.theme.theme === "dark";

  return (
    <ContentWrapper className="gap-10">
      <div
        className={cn("item-center flex min-h-[25rem] justify-between rounded-xl", {
          "bg-gradient-to-l from-[#CFCFCF]  to-[#212121]": userProfile?.theme.theme === "dark",
          "bg-gradient-to-l from-[#3b5ec6] to-[#f5f7fe]": userProfile?.theme.theme === "light",
        })}
      >
        <div className="relative flex flex-col justify-center gap-7 px-14 lg:w-1/2">
          <div className="flex max-w-64 flex-col gap-2">
            <h2 className="text-2xl font-semibold">On-demand snapshots of all your cycles</h2>
            <p className="text-base font-medium text-custom-text-300">
              Monitor cycles across projects, track high-priority issues, and zoom in cycles that need attention.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <a
              className={`${getButtonStyling("primary", "md")} cursor-pointer`}
              href={MARKETING_PRICING_PAGE_LINK}
              target="_blank"
              rel="noreferrer"
            >
              <ProIcon className="h-3.5 w-3.5 text-white" />
              Upgrade
            </a>
          </div>
          <span className="absolute left-0 top-0">
            <Image
              src={`/workspace-active-cycles/cta-l-1-${isDarkMode ? "dark" : "light"}.webp`}
              height={125}
              width={125}
              className="rounded-xl"
              alt="l-1"
            />
          </span>
        </div>
        <div className="relative hidden w-1/2 lg:block">
          <span className="absolute bottom-0 right-0">
            <Image
              src={`/workspace-active-cycles/cta-r-1-${isDarkMode ? "dark" : "light"}.webp`}
              height={420}
              width={500}
              alt="r-1"
            />
          </span>
          <span className="absolute -bottom-16 right-1/2 rounded-xl">
            <Image
              src={`/workspace-active-cycles/cta-r-2-${isDarkMode ? "dark" : "light"}.webp`}
              height={210}
              width={280}
              alt="r-2"
            />
          </span>
        </div>
      </div>
      <div className="grid h-full grid-cols-1 gap-5 pb-8 lg:grid-cols-2 xl:grid-cols-3">
        {WORKSPACE_ACTIVE_CYCLES_DETAILS.map((item) => (
          <div key={item.title} className="flex min-h-32 w-full flex-col gap-2 rounded-md bg-custom-background-80 p-4">
            <div className="flex items-center gap-2 justify-between">
              <h3 className="font-medium">{item.title}</h3>
              <item.icon className="h-4 w-4 text-blue-500" />
            </div>
            <span className="text-sm text-custom-text-300">{item.description}</span>
          </div>
        ))}
      </div>
    </ContentWrapper>
  );
});
