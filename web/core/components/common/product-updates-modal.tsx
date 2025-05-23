"use client";
import React from "react";
import useSWR from "swr";
// headless ui
import { X } from "lucide-react";
import { Dialog, Transition } from "@headlessui/react";
// services
// components
import { Loader } from "@utrack/ui";
import { MarkdownRenderer } from "@/components/ui";
// icons
// helpers
import { renderFormattedDate } from "@/helpers/date-time.helper";
import { WorkspaceService } from "@/utrack-web/services";

type Props = {
  isOpen: boolean;
  setIsOpen: React.Dispatch<React.SetStateAction<boolean>>;
};

// services
const workspaceService = new WorkspaceService();

export const ProductUpdatesModal: React.FC<Props> = ({ isOpen, setIsOpen }) => {
  const { data: updates } = useSWR("PRODUCT_UPDATES", () => workspaceService.getProductUpdates());

  return (
    <Transition.Root show={isOpen} as={React.Fragment}>
      <Dialog as="div" className="relative z-20" onClose={setIsOpen}>
        <Transition.Child
          as={React.Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-custom-backdrop transition-opacity" />
        </Transition.Child>

        <div className="fixed inset-0 z-20 h-full w-full">
          <div className="grid h-full w-full place-items-center p-4">
            <Transition.Child
              as={React.Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
              enterTo="opacity-100 translate-y-0 sm:scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 translate-y-0 sm:scale-100"
              leaveTo="opacity-0 translate-y-4 sm:translate-y-0 sm:scale-95"
            >
              <Dialog.Panel className="relative min-w-[100%] overflow-hidden rounded-lg bg-custom-background-100 shadow-custom-shadow-md sm:min-w-[50%] sm:max-w-[50%]">
                <div className="flex max-h-[90vh] w-full flex-col p-4">
                  <Dialog.Title as="h3" className="flex items-center justify-between text-lg font-semibold">
                    <span>Product Updates</span>
                    <span>
                      <button type="button" onClick={() => setIsOpen(false)}>
                        <X className="h-6 w-6 text-custom-text-200 hover:text-custom-text-100" aria-hidden="true" />
                      </button>
                    </span>
                  </Dialog.Title>
                  {updates && updates.length > 0 ? (
                    <div className="mt-4 h-full space-y-4 overflow-y-auto">
                      {updates.map((item, index) => (
                        <React.Fragment key={item.id}>
                          <div className="flex items-center gap-3 text-xs text-custom-text-200">
                            <span className="flex items-center rounded-full border border-custom-border-200 bg-custom-background-90 px-3 py-1.5 text-xs">
                              {item.tag_name}
                            </span>
                            <span>{renderFormattedDate(item.published_at)}</span>
                            {index === 0 && (
                              <span className="flex items-center rounded-full border border-custom-border-200 bg-custom-primary px-3 py-1.5 text-xs text-white">
                                New
                              </span>
                            )}
                          </div>
                          <MarkdownRenderer markdown={item.body} />
                        </React.Fragment>
                      ))}
                    </div>
                  ) : (
                    <div className="mt-4 grid w-full place-items-center">
                      <Loader className="w-full space-y-6">
                        <div className="space-y-3">
                          <Loader.Item height="30px" />
                          <Loader.Item height="20px" width="80%" />
                          <Loader.Item height="20px" width="80%" />
                        </div>
                        <div className="space-y-3">
                          <Loader.Item height="30px" />
                          <Loader.Item height="20px" width="80%" />
                          <Loader.Item height="20px" width="80%" />
                        </div>
                        <div className="space-y-3">
                          <Loader.Item height="30px" />
                          <Loader.Item height="20px" width="80%" />
                          <Loader.Item height="20px" width="80%" />
                        </div>
                      </Loader>
                    </div>
                  )}
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition.Root>
  );
};
