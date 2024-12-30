"use client";

import { FC } from "react";
import Image from "next/image";
import Link from "next/link";
import { useTheme } from "next-themes";
import { Button } from "@utrack/ui";
// helpers
import { GOD_MODE_URL } from "@/helpers/common.helper";
// images
// assets
import UtrackBackgroundPatternDark from "@/public/auth/background-pattern-dark.svg";
import UtrackBackgroundPattern from "@/public/auth/background-pattern.svg";
import BlackHorizontalLogo from "@/public/utrack-logos/black-horizontal-with-blue-logo.png";
import WhiteHorizontalLogo from "@/public/utrack-logos/white-horizontal-with-blue-logo.png";
import UtrackTakeOffImage from "@/public/utrack-takeoff.png";

export const InstanceNotReady: FC = () => {
  const { resolvedTheme } = useTheme();
  const patternBackground = resolvedTheme === "dark" ? UtrackBackgroundPatternDark : UtrackBackgroundPattern;

  const logo = resolvedTheme === "light" ? BlackHorizontalLogo : WhiteHorizontalLogo;

  return (
    <div className="relative">
      <div className="h-screen w-full overflow-hidden overflow-y-auto flex flex-col">
        <div className="container h-[110px] flex-shrink-0 mx-auto px-5 lg:px-0 flex items-center justify-between gap-5 z-50">
          <div className="flex items-center gap-x-2 py-10">
            <Link href={`/`} className="h-[30px] w-[133px]">
              <Image src={logo} alt="Utrack logo" />
            </Link>
          </div>
        </div>

        <div className="absolute inset-0 z-0">
          <Image src={patternBackground} className="w-screen h-full object-cover" alt="Utrack background pattern" />
        </div>

        <div className="relative z-10 mb-[110px] flex-grow">
          <div className="h-full w-full relative container px-5 mx-auto flex justify-center items-center">
            <div className="w-auto max-w-2xl relative space-y-8 py-10">
              <div className="relative flex flex-col justify-center items-center space-y-4">
                <h1 className="text-3xl font-bold pb-3">Welcome aboard Utrack!</h1>
                <Image src={UtrackTakeOffImage} alt="Utrack Logo" />
                <p className="font-medium text-base text-onboarding-text-400">
                  Get started by setting up your instance and workspace
                </p>
              </div>
              <div>
                <a href={GOD_MODE_URL}>
                  <Button size="lg" className="w-full">
                    Get started
                  </Button>
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
