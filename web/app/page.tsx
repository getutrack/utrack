"use client";

import React, { useEffect } from "react";
import { observer } from "mobx-react";
import Image from "next/image";
import Link from "next/link";
import { motion } from "framer-motion";
// ui
import { useTheme } from "next-themes";
// components
import { AuthRoot } from "@/components/account";
import { PageHead } from "@/components/core";
// constants
import { NAVIGATE_TO_SIGNUP } from "@/constants/event-tracker";
// helpers
import { EAuthModes, EPageTypes } from "@/helpers/authentication.helper";
// hooks
import { useEventTracker } from "@/hooks/store";
// layouts
import DefaultLayout from "@/layouts/default-layout";
// wrappers
import { AuthenticationWrapper } from "@/lib/wrappers";
// assets
import UtrackBackgroundPatternDark from "@/public/auth/background-pattern-dark.svg";
import UtrackBackgroundPattern from "@/public/auth/background-pattern.svg";
import BlackHorizontalLogo from "@/public/utrack-logos/black-horizontal-with-blue-logo.png";
import WhiteHorizontalLogo from "@/public/utrack-logos/white-horizontal-with-blue-logo.png";

const fadeIn = {
  hidden: { opacity: 0, y: 20 },
  visible: { 
    opacity: 1, 
    y: 0,
    transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] }
  }
};

const staggerContainer = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.12
    }
  }
};

const HomePage = observer(() => {
  const { resolvedTheme } = useTheme();
  // hooks
  const { captureEvent } = useEventTracker();

  const logo = resolvedTheme === "light" ? BlackHorizontalLogo : WhiteHorizontalLogo;
  
  useEffect(() => {
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = "auto";
    };
  }, []);

  return (
    <DefaultLayout>
      <AuthenticationWrapper pageType={EPageTypes.NON_AUTHENTICATED}>
        <>
          <div className="relative w-screen h-screen overflow-hidden">
            <PageHead title="Log in - Utrack" />
            
            {/* Background */}
            <div className="absolute inset-0 z-0 bg-gradient-to-br from-primary-50 to-white dark:from-gray-950 dark:to-gray-900">
              <Image
                src={resolvedTheme === "dark" ? UtrackBackgroundPatternDark : UtrackBackgroundPattern}
                className="w-full h-full object-cover opacity-50"
                alt="Utrack background pattern"
              />
            </div>
            
            {/* Content container */}
            <div className="relative z-10 w-screen h-screen overflow-hidden overflow-y-auto flex flex-col">
              
              {/* Navigation header */}
              <motion.div 
                initial={{ opacity: 0, y: -20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
                className="container min-w-full px-4 sm:px-6 md:px-10 lg:px-20 xl:px-36 flex-shrink-0 relative flex items-center justify-between py-6 transition-all"
              >
                <div className="flex items-center gap-x-2">
                  <Link href={`/`} className="h-[30px] w-[133px] relative">
                    <Image src={logo} alt="Utrack logo" className="transition-all duration-300 hover:opacity-90" />
                  </Link>
                </div>
                <div className="flex flex-col items-end sm:items-center sm:gap-2 sm:flex-row text-center text-sm font-medium text-gray-600 dark:text-gray-300">
                  New to Utrack?{" "}
                  <Link
                    href="/sign-up"
                    onClick={() => captureEvent(NAVIGATE_TO_SIGNUP, {})}
                    className="font-semibold text-primary-600 hover:text-primary-700 transition-colors duration-200 hover:underline"
                  >
                    Create an account
                  </Link>
                </div>
              </motion.div>
              
              {/* Main content area */}
              <motion.div 
                variants={staggerContainer}
                initial="hidden"
                animate="visible"
                className="flex flex-col justify-center items-center flex-grow mx-auto w-full max-w-lg px-4 sm:px-6 md:px-10 lg:px-5 transition-all"
              >
                <motion.div 
                  variants={fadeIn} 
                  className="w-full bg-white dark:bg-gray-900 p-8 rounded-xl shadow-lg border border-gray-100 dark:border-gray-800"
                >
                  <motion.div variants={fadeIn} className="mb-6">
                    <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">Welcome back</h1>
                    <p className="text-gray-500 dark:text-gray-400">Log in to continue to your account</p>
                  </motion.div>
                  
                  <motion.div variants={fadeIn}>
                    <AuthRoot authMode={EAuthModes.SIGN_IN} />
                  </motion.div>
                </motion.div>
              </motion.div>
              
              {/* Footer */}
              <motion.div 
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.5, delay: 0.8 }}
                className="py-4 text-center text-sm text-gray-500 dark:text-gray-400"
              >
                Utrack © {new Date().getFullYear()} · Simple, extensible, open-source project management
              </motion.div>
            </div>
          </div>
        </>
      </AuthenticationWrapper>
    </DefaultLayout>
  );
});

export default HomePage;
