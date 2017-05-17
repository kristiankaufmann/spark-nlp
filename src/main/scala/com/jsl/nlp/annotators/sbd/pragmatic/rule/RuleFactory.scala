package com.jsl.nlp.annotators.sbd.pragmatic.rule

import com.jsl.nlp.annotators.sbd.pragmatic.PragmaticSymbols
import com.typesafe.scalalogging.Logger
import org.slf4j.LoggerFactory

import scala.util.matching.Regex

/**
  * Created by Saif Addin on 5/8/2017.
  */
object RuleStrategy extends Enumeration {
  type Strategy = Value
  val APPEND_WITH_SYMBOL,
      PREPEND_WITH_SYMBOL,
      REPLACE_ALL_WITH_SYMBOL,
      REPLACE_WITH_SYMBOL_AND_BREAK,
      PROTECT_WITH_SYMBOL,
      REPLACE_EACH_WITH_SYMBOL,
      REPLACE_EACH_WITH_SYMBOL_AND_BREAK = Value
}

class RuleFactory(ruleStrategy: RuleStrategy.Strategy) {

  import RuleStrategy._

  val logger = Logger(LoggerFactory.getLogger("RuleFactory"))

  private var rules: Seq[Regex] = Seq()
  private var symbolRules: Seq[(String, Regex)] = Seq()

  def addRule(rule: Regex): this.type = {
    rules = rules :+ rule
    this
  }

  def addSymbolicRule(symbol: String, rule: Regex): this.type = {
    symbolRules = symbolRules :+ (symbol, rule)
    this
  }

  def addRules(newRules: Seq[Regex]): this.type = {
    rules = rules ++: newRules
    this
  }

  def applyStrategy(text: String): String = {
    ruleStrategy match {
      case PROTECT_WITH_SYMBOL => rules.foldRight(text)((rule, w) => rule replaceAllIn(w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source}' using rule: '$rule' with strategy $PROTECT_WITH_SYMBOL")
        PragmaticSymbols.PROTECTION_MARKER_OPEN + m.matched + PragmaticSymbols.PROTECTION_MARKER_CLOSE
      }))
      case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
    }
  }

  def applyWith(symbol: String, text: String): String = {
    ruleStrategy match {
      case APPEND_WITH_SYMBOL => rules.foldRight(text)((rule, w) => rule replaceAllIn(w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source}' using rule: '$rule' with strategy $APPEND_WITH_SYMBOL")
        "$0" + symbol
      }))
      case PREPEND_WITH_SYMBOL => rules.foldRight(text)((rule, w) => rule. replaceAllIn(w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source}' using rule: '$rule' with strategy $PREPEND_WITH_SYMBOL")
        symbol + "$0"
      }))
      case REPLACE_ALL_WITH_SYMBOL => rules.foldRight(text)((rule, w) => rule replaceAllIn(w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source}' using rule: '$rule' with strategy $REPLACE_ALL_WITH_SYMBOL")
        symbol
      }))
      case REPLACE_WITH_SYMBOL_AND_BREAK => rules.foldRight(text)((rule, w) => rule. replaceAllIn(
        w, m => {
          logger.debug(s"Matched: '${m.matched}' from: '${m.source}' using rule: '$rule' with strategy $REPLACE_WITH_SYMBOL_AND_BREAK")
          symbol + PragmaticSymbols.BREAK_INDICATOR
        }))
      case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
    }
  }

  def applySymbolicRules(text: String): String = {
    ruleStrategy match {
      case REPLACE_EACH_WITH_SYMBOL => symbolRules.foldRight(text)((rule, w) => rule._2 replaceAllIn(w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source}' using rule: '$rule' with strategy $REPLACE_EACH_WITH_SYMBOL")
        rule._1
      }))
      case REPLACE_EACH_WITH_SYMBOL_AND_BREAK => symbolRules.foldRight(text)((rule, w) => rule._2 replaceAllIn(
        w, m => {
        logger.debug(s"Matched: '${m.matched}' from: '${m.source}' using rule: '$rule' with strategy $REPLACE_EACH_WITH_SYMBOL_AND_BREAK")
        rule._1 + PragmaticSymbols.BREAK_INDICATOR
      }))
      case _ => throw new IllegalArgumentException("Invalid strategy for rule factory")
    }
  }

}