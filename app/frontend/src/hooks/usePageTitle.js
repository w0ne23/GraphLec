import { useEffect } from 'react'

const APP_TITLE = 'GraphLec'

export function formatPageTitle(title = '') {
  const pageTitle = String(title || '').trim()
  return pageTitle ? `${pageTitle} | ${APP_TITLE}` : APP_TITLE
}

export function usePageTitle(title = '') {
  useEffect(() => {
    document.title = formatPageTitle(title)
  }, [title])
}
